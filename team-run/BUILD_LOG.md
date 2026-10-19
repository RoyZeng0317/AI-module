# BUILD LOG

## Deliverables

- Chat: see BUILD_CHAT.md; chats.py, chat_quality.py and regression tests.
- Search: see BUILD_SEARCH.md; tools.py and search tests.
- UI: see BUILD_UI.md; actual backend-served index/style/action and tests.
- Integration: web/backend/app.py returns actual trace string; HTTP tests updated.
- Feature CLI follow-up: missing input image checked before loading model; parser.error instead of traceback. Existing project image successfully exported; 8 classifier tests passed.
- Baseline: BASELINE.json records existing checkpoint CPU greedy outputs for user's three examples, before route repairs.

## Per Sub-task

S1 rules/path verified. No .env inspected or changed, no external AI calls or long training, existing checkpoints retained.
S2/S3: detailed evidence in builder handoffs. 45 chat and 88 tools tests reported; integrated suite pending. English conversational support is finite rules plus existing local generator, not native proficiency.
S4: browser on 127.0.0.1:8081 loaded new real PyScript UI and starter cards; desktop screenshot visually checked. Mobile and send pending. UI tests 2 passed per builder.
S5: 17 targeted HTTP tests passed. Independent review pending. API trace is additive string, local commands trace explicitly local.

## Divergences from Plan

User supplied examples (stock investment ability, typo cross entropy loss, term explanation RAG), then requested English native speakers as quality target. No replacement of self-trained model with cloud models. No general context concatenation because untrained multi-turn format would mislead.
User additionally ran placeholder image path and hit FileNotFoundError; fixed error UX and gave real project path without replacing main task.

## Known Limitations

No claim of general human understanding, Claude/GPT parity, or trained native English. Rules/guards cannot prove arbitrary semantic relevance. Search results are snippets with sources, not verified generated answers; free providers may block/timeout. Existing local models remain limited. Real data training not run.


## Integration update: first review revisions

- REVIEW.md first round found definition-intent precedence MAJOR and bare source URL MINOR; builders addressed both with regression coverage.
- Parent live browser reproduced RAG -> unrelated political profiles. Added lexical rejection and exact reviewed technical glossary (explicit local source/check date, not generation/live); wrong candidate added by this test removed by its exact id 6ebce3ff, preserving pre-existing candidates. User data/pairs untouched.
- Verified live Chinese and English stock ability replies and visible trace details in actual PyScript UI. Verified 390x844 mobile sidebar expand/collapse and readable starter cards; viewport reset afterward. Screenshots frontend-desktop.png, frontend-mobile.png.
- Parent DOM check found #cmdMenu hidden=true yet display:flex height14, causing empty strip; UI builder correcting hidden CSS.
- Initial integrated command named wrong frontend test path and ran zero tests; corrected path yielded 152 passed,14 deselected (excluding detect) before revision; final run pending.


## Final integration evidence

Final combined relevant suite: 176 passed,14 deselected (detect excluded),8.87s; warnings are dependency deprecations/test-stub coroutines. Reviewer independently passed engineering stage,107 directed tests. Subsequent typo wording-only conditional-definition fix revalidated100 search tests. True local evaluation ran24 cases with3 network skips; natural unseen Chinese/English failures remain, not marked semantically passing. No weights changed.
Browser confirmed final RAG definition, English CrossEntropyLoss definition, clickable sources and actual trace; #cmdMenu hidden now display:none,height0. Final screenshot frontend-verified.png. Mobile viewport reset. New preview server on127.0.0.1:8081; user's existing8080 process untouched.

Post-review test isolation: HTTP tests had been writing testclient usage to production ledger. Added autouse temporary usage fixture, restored only testclient baseline while preserving user/runtime keys;20 HTTP tests passed.
