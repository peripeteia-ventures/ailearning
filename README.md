# Latent — AI systems fieldnotes

A personal learning app for senior AI/LLM interviews. React 19, Vite, TypeScript, TanStack Router and Query, Express 5, Tailwind CSS 4, and the existing PostgreSQL 18 installation.

The learning app is running locally at **http://127.0.0.1:3002**. Sign in with **Admin / 123** (username matching is case-insensitive). Private phone access is **https://momentdesktop.tail01307d.ts.net:10001/** once Tailscale Serve is configured. Keep this PC, PostgreSQL, the application process, and Tailscale running. The phone must be connected to the same tailnet.

## UI components

The interface uses shadcn/ui with the Radix Nova style. All 61 available UI component files are installed in `src/components/ui`, with the responsive hook in `src/hooks`. They were downloaded together using the [official CLI's all-components option](https://ui.shadcn.com/docs/cli#add):

```powershell
npx.cmd --yes shadcn@latest add --all --yes
```

`components.json` records the style and `@/` aliases. `src/shadcn.css` maps the component colors to the existing `ThemeProvider`, so the Graphite and Midnight palettes remain centralized. The site uses shadcn buttons, inputs, labels, selects, sliders, cards, badges, tabs, collapsibles, progress meters, loading/error/empty states, tooltips, and the responsive sidebar. Educational SVG diagrams and calculated charts keep their purpose-built renderers.

Local component adjustments pass progress values to Radix for screen readers, label slider thumbs, restore focus to the mobile menu opener, and connect Sonner to the site theme. Preserve these changes when upgrading generated components; review CLI diffs before using `--overwrite`. Desktop navigation stays open, while mobile navigation uses the sidebar's modal Sheet.

## Learning flow

- Browse eight disciplines, search titles and summaries, and filter unread articles, bookmarks, or enrolled decks.
- Read detailed fieldnotes with mathematical examples, responsive diagrams, primary-source citations, pitfalls, and interview scenarios.
- Mark reading progress separately from recall-deck enrollment. Preview individual cards and their study state.
- Review due cards before new cards; reveal the answer, then grade 0–5. Space reveals, number keys grade when focus is outside a control.
- Grades below 4 return for same-session practice. Practice does not change the next due date or scheduled recall accuracy.
- Track coverage, scheduled recall accuracy, review time, activity, future due cards, and recent answers.
- Experiment with KV cache sizing, gradient descent, and offered token workload in the concept lab.

The curriculum covers transformer foundations, optimization and generalization, pretraining data/scaling, post-training alignment, LoRA/QLoRA, attention/GQA, MoE/parallelism, quantization, KV caching, serving schedules, RAG, tools/agents, evaluation, security, three corporate AI farm guides, and a system-design capstone. Each article was assigned to its own Astra content agent with web-search verification and opened primary sources. Sources are embedded in the articles; figures and estimates distinguish calculation from measurement.

**The AI farm is educational content.** The running application does not download models, call LLM providers, or deploy inference containers. The farm's downloadable reference includes Linux Docker Compose, Nginx, two llama.cpp/Qwen inference hosts, two Express replicas, PostgreSQL, and a minimal chat client. Operator-specific model/image verification and configuration are still required. See `examples/ai-farm/README.md` and the three farm articles.

## Start and stop on Windows

Double-click `start.cmd` to start/reuse the app and open a browser. `stop.cmd` stops only the launcher-recorded process after checking its identity. It does not stop PostgreSQL or change any Tailscale routes.

```powershell
Set-Location 'C:\Users\John\Documents\ai learning'
npm.cmd ci
Copy-Item .env.example .env    # initial setup only; preserve an existing .env
npm.cmd run db:setup
npm.cmd run bundle:examples
npm.cmd run build
.\start.cmd
```

`scripts/start.ps1 -NoBrowser` starts quietly. Startup runs the idempotent database seed and rebuilds the frontend; a healthy running instance is reused. `-NoBuild` skips an already-built frontend. After editing frontend code, run `npm.cmd run build`. After editing backend code, stop/start. Content edits require `npm.cmd run db:setup` and a browser refresh.

The launcher records process identity in `work/server-process.json`; logs are `work/server.log` and `work/server-error.log`. The backend binds to loopback. It serves API endpoints and Vite's `dist` output from the same port. Node 24 runs the backend's erasable TypeScript directly; Vite compiles the React frontend.

```powershell
# Development (stop the production server first)
npm.cmd run dev
# Vite: http://127.0.0.1:5174, API: http://127.0.0.1:3002

# Private Tailscale Serve, preserving routes on other HTTPS ports
tailscale serve --bg --https=10001 http://127.0.0.1:3002
tailscale serve status
```

Serve configuration persists; the application must also be running. Neither the launcher nor this build installs a Windows autostart task. Start the app again after reboot. Serve uses a private tailnet route, not Funnel.

## PostgreSQL and identity

The app uses the existing **`ai` database**, with its own **`ailearn` schema**. Configuration is in `.env`; `.env.example` documents the local host, role, and password defaults. No tables in `hw`, `memoize`, or other databases are used or changed.

Tables: `schema_migrations`, `users`, `sessions`, `articles`, `cards`, `user_articles`, `user_cards`, and `reviews`. In psql, use `\dt ailearn.*`, or run `SET search_path TO ailearn, public;` for the current session. The app pool sets its own search path.

Login verifies a salted scrypt password hash and creates a random, opaque, HttpOnly session cookie backed by PostgreSQL; only the hash of the cookie token is stored. Sessions expire after 30 days. Cookies are Secure when accessed through HTTPS. API writes use the session's user identity, parameterized SQL, and explicit allowed browser origins. The `.env` origin list includes the app's local addresses and the Tailscale HTTPS address. Change it and restart when changing the public URL. The default password is intentionally the requested personal-use password; no registration, recovery, SSO, or account-management UI is included.

Article content is stored in PostgreSQL JSONB. Source files in `server/content/*.json` remain the editing source of truth. Seeding upserts articles/cards without clearing study data or resetting Admin's password. New cards are added to already-enrolled decks. Preserve article slugs and card keys when correcting wording. Content deletion/retirement and schema upgrades need deliberate migrations; deleting a JSON file does not delete existing study data. Back up `ai` before destructive changes.

## Review invariants

The scheduler follows SM-2: initial ease 2.5, minimum 1.3; success intervals 1 day, 6 days, then `ceil(previous interval × previous ease)`. Grades 0–2 reset repetitions and schedule 1 day. Ease changes by the SM-2 quality formula. Due dates are exact elapsed 24-hour UTC days, not local calendar midnights; overdue time gives no bonus. Intervals cap at 36,500 days.

Each review transaction locks the user, checks the request UUID before checking the card version, validates eligibility, updates state, and writes history atomically. Identical retries return the stored result. Conflicting UUID reuse or stale card versions return 409. The client preserves its pending UUID after a network failure. Practice updates practice count/history/version only. A fresh queue includes up to 100 due cards plus a selected 0–50 new cards. Refreshing ends the temporary session/practice queue, but committed answers persist.

“Established” means at least 3 consecutive successful scheduled recalls and a 21-day interval. Accuracy uses scheduled attempts, while activity and time include practice. Time can include idle thinking time. Chart days use UTC; due-date displays use the browser's locale. These are study metrics, not certification.

## Verification

```powershell
npm.cmd test
npm.cmd run test:integration
npm.cmd run test:farm-example
node examples/ai-farm/api/test-stream.mjs
npm.cmd run build
```

The application integration suite uses disposable users and removes only those users. It checks authentication, origins, user isolation, enrollment, retries, concurrent review writes, scheduling, practice, reseed preservation, and seeded article payloads. The farm example suite uses a uniquely named temporary schema in `ai`, a fake inference HTTP server, and an ephemeral API process; it cleans up that schema/process. It does **not** test Docker, GPUs, actual llama.cpp, or cloud providers. See `VALIDATION.md` for the recorded checks.

## Changing the theme

All site colors are defined in `src/ThemeContext.tsx`, including surfaces, text, buttons, category markers, charts, walkthroughs, feedback states, and browser branding. The default Graphite preset preserves the existing palette. Set `defaultTheme` to `themes.midnight` in that file to switch to the blue Midnight preset.

To make a custom theme, copy a preset and override its `colors` entries. The `Theme` type checks that every token is present. Components and styles use the corresponding CSS variables, such as `var(--bg)`, `var(--accent)`, and `var(--chart-blue)`; keep literal color values in the theme file.

`ThemeProvider` wraps the entire app, including sign-in and loading states. A component can call `const { theme, setTheme } = useTheme()` and `setTheme(themes.midnight)` to change the palette immediately. `initialTheme` can also be passed to the provider. Runtime choices last until reload; the configured `defaultTheme` is used on the next visit.

## Source map

| Area | Files |
|---|---|
| Typed router and session shell | `src/main.tsx`, `src/Shell.tsx`, `src/api.ts` |
| Learning and articles | `src/Learn.tsx`, `src/ArticlePage.tsx`, `src/Diagram.tsx` |
| Structural visual walkthroughs | `src/VisualLesson.tsx`, `src/visual-lessons.css`, `shared/visuals/` |
| Review and progress | `src/Review.tsx`, `src/Progress.tsx` |
| Interactive experiments | `src/Lab.tsx` |
| Theme, visual system, and responsiveness | `src/ThemeContext.tsx`, `src/styles.css` |
| Express API and persistence | `server/app.ts`, `server/db.ts`, `server/index.ts` |
| Schema, seed, auth, schedule | `server/schema.sql`, `server/setup.ts`, `server/password.ts`, `server/scheduler.ts` |
| Content and types | `server/content/*.json`, `shared/content.ts`, `shared/catalog.ts` |
| Farm reference download | `examples/ai-farm/`, `public/examples/ai-farm.zip` |
| Windows lifecycle | `scripts/start.ps1`, `scripts/stop.ps1`, `start.cmd`, `stop.cmd` |

The supplied `summaryoldanotherproject.md` describes a different app and is preserved as the original reference.

The articles now include 38 original visual walkthroughs with 126 steps across 11 articles, in addition to the existing 120 diagrams. Each transformer foundations section has structural pictures: tiny embedding and Q/K/V matrices, geometric vectors, causal masks, value mixing, head groups, residual paths, normalization, gated MLPs, vocabulary probabilities and cache growth. Related articles visualize gradient descent on a loss surface, training state, efficient attention, quantization, MoE, LoRA/QLoRA, retrieval and speculative decoding. Use Play, Slower, Previous/Next or a numbered step; pictures stay still until requested. Small screens can pan the picture, and every walkthrough has a text equivalent.
