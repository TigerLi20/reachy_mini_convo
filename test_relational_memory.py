import asyncio
import json
import logging
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch, AsyncMock

import numpy as np

from memory_models import MemoryWorker
from memory_runner import DialoguePrivacyFilter, MemoryRealtimeHandler
from relational_memory import RelationalMemory
from relational_store import MemoryStore
from conversation_runtime import ConversationTiming, TimedMedia
from memory_origin import ToolOrigin, tool_origin


class FakeModels:
    def __init__(self):
        self.during_embed = None

    def embed(self, text):
        if self.during_embed:
            callback, self.during_embed = self.during_embed, None
            callback()
        result = np.zeros(4, dtype=np.float32)
        text = text.lower()
        result[0 if any(x in text for x in ['dog', 'milo', 'luna', 'pet']) else 1] = 1
        return result

    def summarize(self, previous, turns):
        return previous + "\n" + " ".join(row['text'] for row in turns if row['role'] == 'user')


class FakeRecognition:
    def __init__(self):
        self.result = {'state': 'known', 'person_id': 'alice', 'name': 'Alice', 'face_count': 1}

    def snapshot(self):
        return dict(self.result)

    def person(self, person):
        self.result = {'state': 'known', 'person_id': person, 'name': person.title(), 'face_count': 1}


class MemoryTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / 'memory.sqlite3'
        self.store = MemoryStore(self.path)
        self.addCleanup(self.store.close)
        self.models = FakeModels()
        self.identity = FakeRecognition()
        self.worker = Mock()
        self.worker.busy.return_value = False
        self.memory = RelationalMemory(self.store, self.identity, self.models, self.worker)
        self.memory.attach_backend()

    def bind(self, person='alice'):
        self.identity.person(person)
        self.memory.tick()
        self.memory.attach_backend()
        self.assertEqual(self.memory.session(False, 10, 'hello')['status'], 'confirmation_required')
        self.assertEqual(self.memory.session(True, 11, 'Yes')['status'], 'bound_restarting')
        self.memory.attach_backend()

    def say(self, text):
        self.memory.speech_started(self.memory.epoch)
        self.memory.transcript(self.memory.epoch, 'user', text)

    def fact(self, value='Milo', key='dog_name'):
        text = f'My dog is called {value}.' if key == 'dog_name' else f'My favorite food is {value}.'
        self.say(text)
        return self.memory.operation('remember', key=key, value=value, quote=text)

    def test_binding_requires_later_affirmation_and_actual_identity(self):
        self.assertEqual(self.memory.operation('review')['status'], 'blocked')
        self.memory.session(False, 1, 'hello')
        for seq, text in [(1, 'Yes'), (2, 'No'), (2, 'Yes but not Alice')]:
            self.assertEqual(self.memory.session(True, seq, text)['status'], 'confirmation_required')
        self.assertIsNone(self.memory.person_id)
        self.identity.result = {'state': 'unknown', 'reason': 'no_confident_match', 'face_count': 1}
        self.assertEqual(self.memory.session(True, 3, 'Yes')['status'], 'temporary_only')

    def test_named_greeting_precedes_confirmation_without_loading_history(self):
        self.bind()
        self.fact()
        fresh = RelationalMemory(self.store, self.identity, self.models, self.worker)
        fresh.attach_backend()
        result = fresh.session(False, 1, 'hello')
        self.assertEqual(result['question'], 'Hi, Alice! Would you like me to open your personal memory?')
        self.assertIsNone(fresh.person_id)
        self.assertNotIn('Milo', json.dumps(fresh.context()))

    def test_distinct_people_keep_conflicting_facts_after_reopening_database(self):
        self.bind('alice')
        self.assertEqual(self.fact('Milo')['status'], 'saved')
        self.bind('bob')
        self.assertEqual(self.fact('Luna')['status'], 'saved')
        with MemoryStoreContext(self.path) as fresh:
            self.assertEqual(fresh.review('alice')[0]['value'], 'Milo')
            self.assertEqual(fresh.review('bob')[0]['value'], 'Luna')
            alice = json.dumps(fresh.recall('alice', 'pet', self.models.embed('pet')))
            bob = json.dumps(fresh.recall('bob', 'pet', self.models.embed('pet')))
            self.assertIn('Milo', alice)
            self.assertNotIn('Luna', alice)
            self.assertIn('Luna', bob)
            self.assertNotIn('Milo', bob)

    def test_group_combinations_pause_private_reads_writes_and_output(self):
        for combination in ['known+known', 'known+unknown', 'unknown+unknown']:
            with self.subTest(combination=combination):
                self.bind('alice')
                self.fact()
                old_epoch = self.memory.epoch
                self.identity.result = {'state': 'unknown', 'reason': 'multiple_faces', 'face_count': 2}
                self.assertFalse(self.memory.output_allowed(old_epoch))
                self.assertIsNone(self.memory.person_id)
                self.assertEqual(self.memory.operation('recall', query='dog')['status'], 'blocked')
                self.assertEqual(self.memory.operation('remember', key='dog_name', value='Wrong', quote='Wrong')['status'], 'blocked')
                self.assertNotIn('Milo', json.dumps(self.memory.context()))
                self.assertEqual(self.memory.session(True, 100, 'Yes')['status'], 'temporary_only')
                self.memory.attach_backend()
                self.assertTrue(self.memory.output_allowed())  # Fresh generic group conversation is allowed.

    def test_new_person_drops_old_context_and_old_speech_capture(self):
        self.bind()
        self.fact()
        epoch = self.memory.epoch
        self.memory.speech_started(epoch)
        self.identity.person('bob')
        self.memory.tick()
        self.memory.transcript(epoch, 'user', 'My bank code is 1234')
        self.assertFalse(self.memory.output_allowed(epoch))
        self.assertNotIn('1234', json.dumps(self.store.turns('alice')))
        self.assertNotIn('Milo', json.dumps(self.memory.context()))

    def test_person_change_during_embedding_does_not_commit_a_fact(self):
        self.bind()
        self.say('My dog is Milo')
        self.models.during_embed = lambda: self.identity.person('bob')
        result = self.memory.operation('remember', key='dog_name', value='Milo', quote='My dog is Milo')
        self.assertEqual(result['status'], 'blocked')
        self.assertEqual(self.store.review('alice'), [])
        self.assertEqual(self.store.review('bob'), [])

    def test_tool_queued_for_previous_person_cannot_adopt_current_person(self):
        self.bind('alice')
        self.say('My dog is Milo')
        origin = ToolOrigin(self.memory.epoch, self.memory.scope(), self.memory.last_user_turn,
                            1, 'My dog is Milo', self.memory.runtime_id)
        media = TimedMedia(Mock(), ConversationTiming(Path(self.directory.name)/'timing.jsonl'))
        media.relational = self.memory
        self.bind('bob')
        self.say('My dog is Milo')  # Even identical text must not allow stale work to cross owners.
        token = tool_origin.set(origin)
        try:
            result = asyncio.run(media.memory_operation('remember', key='dog_name', value='Milo', quote='My dog is Milo'))
            self.assertEqual(result['status'], 'blocked')
            self.assertEqual(asyncio.run(media.memory_session(True))['status'], 'blocked')
        finally:
            tool_origin.reset(token)
        self.assertEqual(self.store.review('bob'), [])

    def test_async_tool_dispatch_carries_scope_and_source_turn_to_worker(self):
        self.bind()
        self.say('My dog is Milo')
        timing = ConversationTiming(Path(self.directory.name)/'timing.jsonl')
        timing.transcript_seq, timing.user_transcript = 1, 'My dog is Milo'
        media = TimedMedia(Mock(), timing)
        media.relational = self.memory
        deps = Mock()
        deps.reachy_mini.media = media
        handler = MemoryRealtimeHandler(deps, self.memory)
        previous = tool_origin.get()

        async def run_tool():
            handler._mark_activity('tool_call_received')
            return await asyncio.create_task(media.memory_operation('remember', key='dog_name', value='Milo', quote='My dog is Milo'))
        try:
            self.assertEqual(asyncio.run(run_tool())['status'], 'saved')
        finally:
            tool_origin.set(previous)
        self.assertEqual(self.store.review('alice')[0]['value'], 'Milo')

    def test_a_later_turn_does_not_replace_the_original_tool_source(self):
        self.bind()
        self.say('My dog is Milo')
        original_source = self.memory.last_user_turn
        self.models.during_embed = lambda: self.say('My favorite food is ramen')
        result = self.memory.operation('remember', key='dog_name', value='Milo', quote='My dog is Milo')
        self.assertEqual(result['status'], 'saved')
        self.assertEqual(self.store.review('alice')[0]['source_turn'], original_source)

    def test_pending_farewell_cannot_sleep_a_changed_conversation(self):
        self.bind()
        origin = ToolOrigin(self.memory.epoch, self.memory.scope(), None, 1, 'bye', self.memory.runtime_id)
        timing = Mock()
        media = TimedMedia(Mock(), timing)
        media.relational = self.memory
        callback = Mock()
        token = tool_origin.set(origin)
        try:
            asyncio.run(media.finish_conversation(callback))
        finally:
            tool_origin.reset(token)
        guarded = timing.request_farewell.call_args.args[0]
        self.identity.person('bob')
        self.assertEqual(guarded()['status'], 'cancelled_context_changed')
        callback.assert_not_called()

    def test_confirmed_session_survives_dropout_then_expires_once_at_20_seconds(self):
        for scene in [{'state': 'unknown', 'reason': 'no_face', 'face_count': 0},
                      {'state': 'unknown', 'reason': 'no_confident_match', 'face_count': 1}]:
            with self.subTest(scene=scene), patch('relational_memory.monotonic', return_value=100) as clock:
                self.bind()
                original_scope = self.memory.scope()
                self.identity.result = scene
                self.memory.tick()
                clock.return_value = 119.9
                self.assertEqual(self.memory.scope(), original_scope)
                self.assertIn(self.fact()['status'], ('saved', 'unchanged'))
                self.assertEqual(self.memory.session(False, 20, 'hello')['status'], 'bound')
                self.assertTrue(self.memory.output_allowed(original_scope.epoch))
                clock.return_value = 120
                self.memory.tick()
                self.assertIsNone(self.memory.person_id)
                self.assertEqual(self.memory.notice, 'identity_lost')
                self.assertFalse(self.memory.output_allowed(original_scope.epoch))
                self.assertEqual(self.memory.operation('review')['status'], 'blocked')
                self.assertNotIn('Milo', json.dumps(self.memory.context()))
                ended_epoch = self.memory.attach_backend()
                clock.return_value = 150
                self.memory.tick()
                self.assertEqual(self.memory.epoch, ended_epoch)  # No repeated adjustment prompt/restart.

    def test_reacquisition_preserves_session_and_restarts_loss_timer(self):
        with patch('relational_memory.monotonic', return_value=100) as clock:
            self.bind()
            original_scope = self.memory.scope()
            for start in [100, 130]:
                clock.return_value = start
                self.identity.result = {'state': 'unknown', 'reason': 'no_face', 'face_count': 0}
                self.memory.tick()
                clock.return_value = start + 19
                self.identity.person('alice')
                self.assertEqual(self.memory.scope(), original_scope)
                self.assertIsNone(self.memory.uncertain_at)

    def test_group_or_different_person_during_grace_revokes_session_immediately(self):
        for scene in [{'state': 'unknown', 'reason': 'multiple_faces', 'face_count': 2},
                      {'state': 'known', 'person_id': 'bob', 'name': 'Bob', 'face_count': 1}]:
            with self.subTest(scene=scene), patch('relational_memory.monotonic', return_value=100) as clock:
                self.bind()
                epoch = self.memory.epoch
                self.identity.result = {'state': 'unknown', 'reason': 'no_face', 'face_count': 0}
                self.memory.tick()
                clock.return_value = 101
                self.identity.result = scene
                self.assertFalse(self.memory.output_allowed(epoch))
                self.assertIsNone(self.memory.person_id)
                self.assertIsNone(self.memory.confirmed_identity())

    def test_identity_tool_keeps_confirmed_name_during_grace_without_waiting(self):
        with patch('relational_memory.monotonic', return_value=100):
            self.bind()
            self.identity.result = {'state': 'unknown', 'reason': 'no_face', 'face_count': 0}
            media = TimedMedia(Mock(), ConversationTiming(Path(self.directory.name)/'timing.jsonl'))
            media.relational, media.recognition = self.memory, self.identity
            origin = ToolOrigin(self.memory.epoch, self.memory.scope(), None, 1, 'hello', self.memory.runtime_id)
            token = tool_origin.set(origin)
            try:
                with patch('conversation_runtime.asyncio.sleep', new=AsyncMock()) as wait:
                    result = asyncio.run(media.person_identity())
                    wait.assert_not_awaited()
                self.assertEqual(result['state'], 'session_confirmed')
                self.assertEqual(result['name'], 'Alice')
            finally:
                tool_origin.reset(token)

    def test_explicit_identity_correction_ends_grace_immediately(self):
        self.bind()
        self.identity.result = {'state': 'unknown', 'reason': 'no_face', 'face_count': 0}
        media = TimedMedia(Mock(), ConversationTiming(Path(self.directory.name)/'timing.jsonl'))
        media.relational, media.recognition = self.memory, self.identity
        self.identity.reject_current = Mock(return_value={'state': 'unknown', 'reason': 'identity_rejected'})
        origin = ToolOrigin(self.memory.epoch, self.memory.scope(), None, 1, 'Not Alice', self.memory.runtime_id)
        token = tool_origin.set(origin)
        try:
            asyncio.run(media.person_identity(reject_current=True))
        finally:
            tool_origin.reset(token)
        self.assertIsNone(self.memory.person_id)
        self.assertIsNone(self.memory.scope())
        self.assertEqual(self.memory.notice, 'identity_rejected')

    def test_source_quote_cannot_be_fabricated_or_belong_to_another_person(self):
        self.bind()
        self.say('I have a dog')
        self.assertEqual(self.memory.operation('remember', key='dog_name', value='Milo', quote='My dog is Milo')['status'], 'not_saved')
        self.assertEqual(self.store.review('alice'), [])
        bob = self.store.start('bob')
        source = self.store.append('bob', bob, 'user', 'My dog is Luna')
        with self.assertRaises(ValueError):
            self.store.remember('alice', self.memory.session_id, source, 'dog_name', 'Luna', 'My dog is Luna')
        self.say('Bob has a dog called Luna')
        self.assertEqual(self.memory.operation('remember', key='dog_name', value='Luna', quote='Bob has a dog called Luna')['status'], 'not_saved')
        self.say('My friend Bob has a dog called Luna')
        self.assertEqual(self.memory.operation('remember', key='dog_name', value='Luna', quote='My friend Bob has a dog called Luna')['status'], 'not_saved')

    def test_questions_and_hypotheticals_do_not_relearn_forgotten_facts(self):
        self.bind()
        for text in ['Was my dog called Milo?', 'If I had a dog, I would call it Milo.']:
            self.say(text)
            result = self.memory.operation('remember', key='dog_name', value='Milo', quote=text)
            self.assertEqual(result['status'], 'not_saved')
        self.assertEqual(self.store.review('alice'), [])

    def test_relational_profile_and_real_session_configuration_load_in_a_fresh_process(self):
        root = Path(__file__).resolve().parent
        env = os.environ.copy()
        env.update({'REACHY_MINI_EXTERNAL_PROFILES_DIRECTORY': str(root / 'profiles'),
                    'REACHY_MINI_CUSTOM_PROFILE': 'relational_conversation',
                    'REACHY_MINI_EXTERNAL_TOOLS_DIRECTORY': str(root / 'conversation_tools'),
                    'AUTOLOAD_EXTERNAL_TOOLS': 'false', 'XDG_DATA_HOME': self.directory.name})
        code = '''
import sys
from unittest.mock import Mock
from pathlib import Path
from reachy_mini_conversation_app.tools.core_tools import initialize_tools,get_tools,ToolDependencies
from memory_runner import MemoryRealtimeHandler
from relational_store import MemoryStore
from relational_memory import RelationalMemory
initialize_tools(sys.argv[1])
assert set(get_tools()) == {'finish_conversation','person_identity','enroll_person','memory_session','relational_memory','task_status','task_cancel'}
db=MemoryStore(Path(sys.argv[1])/'fresh.sqlite3')
worker=Mock(); worker.busy.return_value=False
memory=RelationalMemory(db,None,Mock(),worker)
handler=MemoryRealtimeHandler(ToolDependencies(Mock(),Mock()),memory,sys.argv[1])
config=handler._get_session_config([tool.spec() for tool in get_tools().values()])
assert 'MEMORY CONTEXT' in config['instructions']
assert 'temporary' in config['instructions']
assert config['type']=='realtime'
db.close()
'''
        result = subprocess.run([sys.executable, '-c', code, self.directory.name], env=env, cwd=root,
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_episode_deletion_confirms_cascade_and_stays_with_its_owner(self):
        self.bind()
        self.fact()
        self.memory.finish()
        job = self.store.summary_input('alice', self.memory.session_id, final=True)
        self.store.save_summary(job, 'My dog is Milo', self.models.embed('dog'))
        episode = self.store.recall('alice', 'dog', self.models.embed('dog'))
        episode_id = next(entry['id'] for entry in episode if entry['kind'] == 'episode')
        self.say('Forget that episode')
        self.assertEqual(self.memory.operation('forget_episode', query=episode_id)['status'], 'confirmation_required')
        self.assertEqual(self.memory.operation('forget_episode', query=episode_id, confirmed=True)['status'], 'confirmation_required')
        self.say('Yes, please')
        result = self.memory.operation('forget_episode', query=episode_id, confirmed=True)
        self.assertEqual(result['status'], 'forgotten_episode')
        self.assertEqual(self.store.review('alice'), [])
        self.assertNotIn('Milo', json.dumps(self.store.turns('alice')))
        self.assertEqual(self.store.forget_episode('bob', episode_id)['status'], 'not_found')

    def test_correction_removes_old_value_from_dialogue_summary_search_and_context(self):
        self.bind()
        self.fact()
        alice_session = self.memory.session_id
        job = self.store.summary_input('alice', alice_session, final=True)
        self.store.save_summary(job, 'The dog is Milo', self.models.embed('dog'))
        self.say('Actually, my dog is Max, not Milo.')
        result = self.memory.operation('correct', key='dog_name', value='Max', quote='Actually, my dog is Max, not Milo.')
        self.assertEqual(result['status'], 'corrected')
        self.assertFalse(self.memory.output_allowed())
        self.memory.attach_backend()
        context = json.dumps(self.memory.context())
        self.assertIn('Max', context)
        self.assertNotIn('Milo', context)
        self.assertFalse(self.store.save_summary(job, 'Milo could return', self.models.embed('dog')))
        self.assertNotIn('Milo', json.dumps(self.store.recall('alice', 'dog', self.models.embed('dog'))))

    def test_fact_deletion_removes_related_copies_without_affecting_other_person(self):
        self.bind()
        self.fact()
        self.fact('pasta', 'favorite_food')
        alice_session = self.memory.session_id
        job = self.store.summary_input('alice', alice_session, final=True)
        self.store.save_summary(job, 'Milo is a dog', self.models.embed('dog'))
        bob = self.store.start('bob')
        source = self.store.append('bob', bob, 'user', 'My dog is Milo')
        self.store.remember('bob', bob, source, 'dog_name', 'Milo', 'My dog is Milo')
        result = self.memory.operation('forget', key='dog_name')
        self.assertEqual(result['status'], 'forgotten')
        self.memory.attach_backend()
        self.assertNotIn('Milo', json.dumps(self.memory.context()))
        self.assertIn('pasta', json.dumps(self.memory.context()))
        self.assertEqual(self.store.review('bob')[0]['value'], 'Milo')
        self.assertFalse(self.store.save_summary(job, 'Milo is a dog'))

    def test_forget_all_requires_actual_request_and_preserves_other_users_and_identity(self):
        self.bind()
        self.fact()
        self.assertEqual(self.memory.operation('forget_all')['status'], 'not_saved')
        bob = self.store.start('bob')
        self.store.append('bob', bob, 'user', 'I like tea')
        self.say('Please forget all our conversations')
        result = self.memory.operation('forget_all')
        self.assertEqual(result['status'], 'forgotten_all')
        self.assertTrue(result['face_enrollment_retained'])
        self.assertEqual(self.store.review('alice'), [])
        self.assertEqual(self.store.turns('alice'), [])
        self.assertEqual(len(self.store.turns('bob')), 1)

    def test_eight_turns_roll_context_only_after_reply_and_consolidation(self):
        self.bind()
        old_epoch = self.memory.epoch
        for i in range(8):
            self.say(f'My topic number is {i}')
            if i < 7:
                self.memory.transcript(old_epoch, 'assistant', 'Tell me more')
        self.memory.compact_if_ready()
        self.assertEqual(self.memory.epoch, old_epoch)
        self.memory.transcript(old_epoch, 'assistant', 'Thanks for telling me')
        self.worker.busy.return_value = True
        self.memory.compact_if_ready()
        self.assertEqual(self.memory.epoch, old_epoch)
        self.worker.busy.return_value = False
        self.memory.compact_if_ready()
        self.assertGreater(self.memory.epoch, old_epoch)
        self.memory.attach_backend()
        context = self.memory.context()
        self.assertLessEqual(len(context['recent_dialogue']), 16)
        self.assertEqual(self.memory.person_id, 'alice')

    def test_context_budget_is_bounded_and_other_person_cannot_enter_it(self):
        self.bind()
        for _ in range(20):
            self.say('x' * 10000)
        bob = self.store.start('bob')
        self.store.append('bob', bob, 'user', 'BOB_PRIVATE_SENTINEL')
        context = json.dumps(self.memory.context())
        self.assertLessEqual(len(context), 18000)
        self.assertNotIn('BOB_PRIVATE_SENTINEL', context)

    def test_summary_worker_produces_scoped_retrievable_episode(self):
        session = self.store.start('alice')
        self.store.append('alice', session, 'user', 'My dog Milo and I went to the park')
        worker = MemoryWorker(self.store, self.models)
        worker.submit('alice', session, final=True)
        self.assertTrue(worker.close())
        episodes = self.store.recall('alice', 'pet', self.models.embed('pet'))
        self.assertEqual(episodes[0]['kind'], 'episode')
        self.assertIn('Milo', episodes[0]['text'])
        self.assertEqual(self.store.recall('bob', 'pet', self.models.embed('pet')), [])

    def test_old_episode_is_retained_after_later_running_summary(self):
        session = self.store.start('alice')
        self.store.append('alice', session, 'user', 'My dog Milo visited the park')
        first = self.store.summary_input('alice', session, final=True)
        self.store.save_summary(first, 'My dog Milo visited the park', self.models.embed('dog'))
        self.store.append('alice', session, 'user', 'My favorite food is pasta')
        second = self.store.summary_input('alice', session, final=True)
        self.store.save_summary(second, 'My favorite food is pasta', self.models.embed('food'))
        recall = self.store.recall('alice', 'dog', self.models.embed('dog'))
        self.assertTrue(any('Milo' in entry['text'] for entry in recall))

    def test_committed_dialogue_survives_abrupt_process_exit(self):
        path = Path(self.directory.name) / 'crash.sqlite3'
        code = "from pathlib import Path; from relational_store import MemoryStore; import os,sys; s=MemoryStore(Path(sys.argv[1])); session=s.start('alice'); s.append('alice',session,'user','Crash survivor'); os._exit(0)"
        subprocess.run([sys.executable, '-c', code, str(path)], check=True)
        with MemoryStoreContext(path) as fresh:
            self.assertEqual(fresh.turns('alice')[0]['text'], 'Crash survivor')
            self.assertEqual(len(fresh.recoverable_sessions()), 1)

    def test_privacy_filter_removes_transcripts_tool_payloads_and_tracebacks(self):
        for name, text in [('reachy_mini_conversation_app.console', 'role=user content=SECRET'),
                           ('reachy_mini_conversation_app.huggingface_realtime', 'Tool args=SECRET')]:
            record = logging.LogRecord(name, logging.INFO, '', 1, text, (), None)
            DialoguePrivacyFilter().filter(record)
            self.assertNotIn('SECRET', record.getMessage())

    def test_old_handler_output_is_dropped_after_identity_changes(self):
        self.bind()
        deps = Mock()
        handler = MemoryRealtimeHandler(deps, self.memory)
        self.identity.person('bob')
        with patch('memory_runner.HuggingFaceRealtimeHandler.emit', new=AsyncMock(return_value=(16000, np.ones(10)))):
            self.assertIsNone(asyncio.run(handler.emit()))

    def test_sleep_shutdown_cannot_restart_context_as_camera_leaves_view(self):
        self.bind()
        self.fact()
        epoch = self.memory.epoch
        self.memory.closing = True
        self.identity.result = {'state': 'unknown', 'reason': 'no_face', 'face_count': 0}
        self.memory.tick()
        self.memory.finish()
        self.assertEqual(self.memory.epoch, epoch)
        self.assertFalse(self.memory.output_allowed())
        self.assertIsNone(self.memory.scope())

    def test_backend_replacement_contains_only_the_confirmed_persons_context(self):
        self.bind('alice')
        self.fact('Milo')
        alice = MemoryRealtimeHandler(Mock(), self.memory)
        with patch('memory_runner.HuggingFaceRealtimeHandler._get_session_config', return_value={'instructions': 'base'}):
            config = alice._get_session_config([])
        self.assertIn('Milo', config['instructions'])
        self.assertNotIn('Luna', config['instructions'])
        self.bind('bob')
        self.fact('Luna')
        bob = MemoryRealtimeHandler(Mock(), self.memory)
        with patch('memory_runner.HuggingFaceRealtimeHandler._get_session_config', return_value={'instructions': 'base'}):
            config = bob._get_session_config([])
        self.assertIn('Luna', config['instructions'])
        self.assertNotIn('Milo', config['instructions'])
        self.assertNotEqual(alice.memory_epoch, bob.memory_epoch)
        self.assertFalse(self.memory.output_allowed(alice.memory_epoch))

    def test_reconnect_audio_is_discarded_when_its_person_epoch_expires(self):
        self.bind()
        handler = MemoryRealtimeHandler(Mock(), self.memory)
        asyncio.run(handler.receive((16000, np.ones((10, 1), dtype=np.float32))))
        self.assertEqual(len(handler.waiting_audio), 1)
        self.identity.person('bob')
        asyncio.run(handler.receive((16000, np.ones((10, 1), dtype=np.float32))))
        self.assertEqual(len(handler.waiting_audio), 0)

    def test_old_tool_result_never_reaches_new_persons_backend(self):
        self.bind()
        handler = MemoryRealtimeHandler(Mock(), self.memory)
        result = Mock()
        result.result = {'facts': ['PRIVATE'], '_memory_epoch': self.memory.epoch}
        self.identity.person('bob')
        self.memory.tick()
        with patch('memory_runner.HuggingFaceRealtimeHandler._handle_tool_result', new=AsyncMock()) as parent:
            asyncio.run(handler._handle_tool_result(result))
            parent.assert_not_awaited()


class MemoryStoreContext:
    def __init__(self, path):
        self.store = MemoryStore(path)

    def __enter__(self):
        return self.store

    def __exit__(self, *args):
        self.store.close()


if __name__ == '__main__':
    unittest.main()
