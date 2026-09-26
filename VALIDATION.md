# Validation record — 2026-09-25

## Structural visual learning update — 2026-09-25

Added 38 original SVG walkthroughs containing 126 steps across 11 articles. Every section in transformer foundations now has a structural visual. Existing prose, 120 diagrams and 252 recall cards are retained; article JSON changes add paragraph-level placement metadata. No image-generation assets or new dependencies are required.

The player supports explicit play/pause/replay, reset, numbered steps, previous/next, slower timing, accessible SVG descriptions and a text transcript. It stops at the final frame, pauses outside the viewport or in a hidden document, and respects reduced motion with manual stepping. At narrow widths, the 600-pixel minimum picture can be panned while narration and controls reflow.

Verification completed for this update:

- `npm.cmd run build`: TypeScript and Vite production build passed after final changes.
- `npm.cmd test`: all 10 tests passed. New checks verify placement references, coverage of foundations sections, distinct steps, unique IDs, matrix bounds/labels, finite values, and attention/gradient arithmetic.
- `npm.cmd run test:integration`: passed against PostgreSQL, including reseed preservation and all 18 article payloads. Disposable integration users removed. Current article placements were seeded into the running app.
- In-app browser: traversed all 126 steps at desktop widths (1118 and 1280 CSS pixels across checks). Checked rendered text bounds and overlap, and inspected representative decoder, Q/K/V, SwiGLU, RoPE, QLoRA, gradient-descent and speculative-decoding pictures. Corrected tight loss-axis spacing, an arrow/label intersection and transient matrix-text crossings. Final foundations/training frame audits are clean; the other article audits were clean after mathematical fixes.
- All 11 affected article pages checked at a measured 390 CSS-pixel viewport: no page-wide horizontal overflow or clipped playback/step controls. Verified the diagram's own horizontal scroll, keyboard Enter stepping, playback completion, reset, pause, slower checkbox and expanded text equivalent. Temporary viewport override reset.
- Independent Astra review corrected KV-query ownership, RoPE vector lengths and the additive QLoRA branch. Tiny numerical examples were reviewed against article primary sources. Reduced-motion behavior was reviewed in code; an OS-level preference change and physical-phone testing were not performed.
- Browser QA used a disposable user, which was removed afterward. No real learning progress was changed. Existing app server remained running; production static assets were rebuilt in place. Unrelated infrastructure and farm example code were not changed or retested.

Authoring instructions and source-map documentation now cover visual placements and scene data. The scene registry is `shared/visuals/index.ts`; the reusable player is `src/VisualLesson.tsx`.

## Delivered

- Working React/Vite/TypeScript frontend with TanStack Router and Query, Tailwind CSS, Express, and the existing PostgreSQL `ai` database (`ailearn` schema).
- 8 disciplines, 18 articles, 120 diagrams, and 252 flashcards. Every article has 6–9 diagrams and 14 cards. The corpus contains about 32,000 words of section prose, plus figures, examples, prompts, and references.
- One Astra content-author task per article; authors searched the web and opened primary/official sources. Citations and qualification of illustrative numbers are embedded in the content.
- Admin / 123 login, PostgreSQL-backed sessions, search, bookmarks, reading state, deck previews/enrollment, SM-2 reviews with reinforcement, progress charts/history, and three interactive concept experiments.
- Three corporate AI farm guides and an 11-file downloadable reference bundle. The farm remains educational content, separate from the running learning app.

## Automated checks passed

`npm.cmd test`: 7 checks passed. Complete unique curriculum/prerequisite links; article depth, sources, and recall prompts; diagram structure and finite coordinates; SM-2 success intervals, failure/lapse behavior, practice isolation, grade validation and interval ceiling.

`npm.cmd run test:integration`: passed against real PostgreSQL. Health, authentication, invalid credentials, allowed/disallowed browser origins, session identity, per-user progress isolation, enrollment idempotency, reading independent of enrollment, first reviews, exact retry replay, conflicting request reuse, stale versions, early-review rejection, cross-user card access rejection, practice without rescheduling, two concurrent writes with only one accepted, invalid grades, stats, seed preservation, all 18 article/deck payloads, and logout. Disposable integration users were removed.

`npm.cmd run test:farm-example`: passed the teaching Express reference using a uniquely named temporary PostgreSQL schema, an ephemeral local API process, and a deterministic fake inference endpoint. Checked opaque login, streamed durable text, terminal snapshot, replay without another inference call, revision conflict, cloud disabled by default, truncated stream becoming partial, expired lease recovery, and ownership checks. Temporary schema and processes were removed. No real model/provider calls.

`node examples/ai-farm/api/test-stream.mjs`: passed fragmented UTF-8, CRLF boundaries, multiline SSE data, incomplete EOF, and oversized-frame checks. Server/client JavaScript syntax checks were also performed by the article author.

`npm.cmd run build`: TypeScript check and production Vite build passed. Article, review, progress, and lab screens use separate route chunks. The initial dependency installation reported zero npm audit vulnerabilities.

## Browser checks passed

Used the actual running production app through the Codex browser. A disposable browser QA user exercised sign-in/sign-out, article navigation, bookmarking, mark-as-explored, enrollment, card previews, five scheduled answers including one missed answer, the resulting practice repeat, completion summary, progress, mobile menu, search, and concept-lab controls. The session displayed 5 scheduled answers, 80% recall, and 1 practice answer. KV head change produced 0.50 GiB; the first gradient step at η=0.2 produced w=2.4 and loss=5.76.

Inspected all 18 articles at a measured 390 CSS-pixel viewport (378 content pixels with the desktop browser's scrollbar) and at a measured 1200 CSS-pixel viewport. All 120 diagram containers fit; no final document horizontal overflow. The deployment article initially overflowed on an unbroken SHA-256 digest; word wrapping was added and its mobile check passed. Code blocks intentionally scroll within their own containers. Representative login, dashboard, article, and mobile review layouts were visually inspected. No browser warning/error logs were present in the inspected final article session. This is targeted browser QA, not a formal accessibility audit or a physical-phone test.

Restarted the hidden production process through the supplied stop/start scripts, reseeded, and verified the browser QA user's session and saved 1/18 reading progress, 80% scheduled recall, and history persisted. The browser QA user was subsequently removed. Admin's learning state was left untouched (zero article-progress rows and zero reviews at verification).

## Private HTTPS verified

Tailscale Serve runs in background:

`https://momentdesktop.tail01307d.ts.net:10001/` → `http://127.0.0.1:3002`

Existing HTTPS routes on 443, 8443, and 10000 remain unchanged. Host-network HTTPS checks validated the certificate normally, health response, Admin / 123 login, authenticated catalog (18 articles / 120 diagrams / 252 cards), and a 200 response for the 18,964-byte example zip. The archive was inspected and contains the intended `.env.example`, README, API/client, stream tests, Dockerfile, and Compose/Nginx files. No real `.env` or dependency directory is bundled.

The final browser also signed in successfully through the Tailscale URL and displayed the full catalog. Its temporary responsive viewport override was reset. Phone access still requires the user's phone to be connected to the tailnet; the user will perform that physical-device check.

## Scope limits

No Docker containers, GPUs, llama.cpp binaries, or model files were installed or deployed for the farm guides. No OpenAI or Anthropic call was made. Docker/network/GPU compatibility, throughput, cloud-account permissions, and real-provider streaming remain operator validation steps. The farm's immutable model artifact and selected image digest must be verified on deployment; the guide records source-fetch limitations explicitly. The reference is a teaching scaffold with documented shared Admin identity, replica-local admission, fixed leases, character-based prompt limits, and no enterprise HA/SSO system.

This learning app runs as a hidden process, not a Windows boot service. Use `start.cmd` after a reboot. Keep the PC, PostgreSQL, app, and Tailscale running for phone access. The original reference project and its database/Serve route were preserved.
