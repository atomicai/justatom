# Universe Retrieval Corpus: Living Wiki

This is the durable historical record for extending the JustAtom retrieval
corpus to 10,000 passages. Immutable run outcomes remain in
`universe-retrieval-runs.tsv`; local `llms.txt` development memory is not
versioned.

## Fixed contract

- Git distribution: metadata and loader integration only; no corpus payload
- Full canonical benchmark: private Hugging Face
  `justatom/universe-retrieval-benchmark`, config/split `benchmark/full`
- Baseline: 5,292 passages, 15,624 queries
- Baseline SHA-256: `3a8e0d0b4179ab0252a8e5f3589f02bf393c24380f35b8db34df0a424d3ed8b8`
- Final: 10,000 passages, 43,872 queries
- Cleaned final SHA-256 after image-layer removal:
  `323cdffbab56ba6d0fbee62e3d465ad859a191122bf38eebd52951afc020b27d`
- Retrieval benchmark manifest:
  `docs/datasets/universe-retrieval-benchmark-manifest.json`
- Harry Potter addition: 1,600 passages
- Hunger Games addition: 1,300 passages
- Middle-earth addition: 1,808 passages
- Scratch root: `/private/tmp/justatom-universe-retrieval-20260825`
- Model snapshot: `gpt-5.6-sol`; no aliases and no fallback models
- Generation reasoning: `medium`
- Audit and repair reasoning: `high`
- Every new passage: 3 EN + 3 RU questions, 4–8 EN + 4–8 RU keyphrases
- Canonical rows never contain `license` or `license_url`

## Source endpoints

| Universe | MediaWiki API | Canonical article base |
| --- | --- | --- |
| Harry Potter | `https://harrypotter.fandom.com/api.php` | `https://harrypotter.fandom.com/wiki/` |
| The Hunger Games | `https://thehungergames.fandom.com/api.php` | `https://thehungergames.fandom.com/wiki/` |
| Middle-earth | `https://tolkiengateway.net/w/api.php` | `https://tolkiengateway.net/wiki/` |

Use namespace-0 pages and `action=parse`, `prop=text|categories|revid`,
`redirects=1`, `format=json`, `formatversion=2`. Preserve page/revision IDs,
canonical URL, retrieval timestamp, section path, and section-local index.

## Selection rules

Selection finishes before generation. Remove infoboxes, tables, references,
navigation, galleries, figures, scripts, styles, edit controls, and citation
markers. Reject disambiguation pages, fanon, lists without explanatory prose,
trivia, appearances, references, external links, merchandise, role-playing,
game-stat tables, unresolved-pronoun openings, incomplete sentences, and
chunks outside 420–1,800 characters.

Normalized exact duplicates and 5-word-shingle Jaccard pairs at or above
0.78 cannot both survive. Harry Potter and Middle-earth keep at most four
chunks per page. The compact Hunger Games wiki needs a cap of 15 to retain the
full 10% reserve after all quality filters; its final 1,300 rows still cover
390 distinct pages. No entity class may exceed 35% of a frozen universe.

Raw MediaWiki categories are discovery and classification signals. They are
not retrieval keywords. GPT-5.6 Sol generates passage-grounded keyphrases
only after a passage manifest is frozen.

## Reviewed discovery seeds and category allowlists

Seeds guarantee cornerstone coverage; they do not bypass passage gates. Exact
MediaWiki redirects are resolved before extraction. Missing titles are logged
and replaced only by eligible namespace-0 articles from the reviewed category
graph or the longest eligible article inventory.

### Harry Potter seeds

- Characters: Harry Potter, Hermione Granger, Ron Weasley, Albus Dumbledore,
  Lord Voldemort, Severus Snape, Rubeus Hagrid, Sirius Black, Remus Lupin,
  Minerva McGonagall, Draco Malfoy, Ginny Weasley, Dobby.
- Places: Hogwarts School of Witchcraft and Wizardry, Diagon Alley, Hogsmeade,
  Ministry of Magic, Azkaban, Forbidden Forest, Godric's Hollow, Gringotts
  Wizarding Bank.
- Organizations: Order of the Phoenix, Death Eaters, Dumbledore's Army,
  Wizengamot, Auror Office, Hogwarts Houses.
- Objects: Elder Wand, Resurrection Stone, Cloak of Invisibility, Marauder's
  Map, Horcrux, Philosopher's Stone, Sorting Hat, Time-Turner.
- Events: Battle of Hogwarts, First Wizarding War, Second Wizarding War,
  Triwizard Tournament, opening of the Chamber of Secrets, Yule Ball.
- Creatures: Dementor, house-elf, basilisk, phoenix, hippogriff, dragon.
- Systems: magic, Patronus Charm, Unforgivable Curses, Polyjuice Potion,
  Quidditch, Apparition, Occlumency, Transfiguration.
- Works/adaptations: Harry Potter novels, Harry Potter films, Fantastic Beasts
  film series, Hogwarts Legacy.

Allowed category concepts: characters/people/students, locations/settlements,
organizations/government/groups, objects/artifacts/weapons, battles/wars/events,
creatures/species, spells/charms/curses/potions/magical disciplines, novels/
films/games/adaptations, and factual adaptation-production biographies. Reject
fanon, role-playing, merchandise, image-only, list, template, and community
pages. Fictional in-universe newspaper transcripts remain eligible when the
chunk is self-contained factual prose rather than lyrics, questions, or a
script fragment.

### Hunger Games seeds

- Characters: Katniss Everdeen, Peeta Mellark, Gale Hawthorne, Haymitch
  Abernathy, Effie Trinket, Primrose Everdeen, Coriolanus Snow, Alma Coin,
  Finnick Odair, Johanna Mason, Cinna, Plutarch Heavensbee, Rue, Beetee,
  Wiress, Lucy Gray Baird.
- Places: Panem, the Capitol, District 12, District 13, District 2, Victor's
  Village, Training Center, the arenas.
- Organizations: Peacekeepers, Gamemakers, Career Tributes, rebels, Covey.
- Objects: mockingjay pin, Katniss's bow and arrows, nightlock berries,
  tracker, parachute, propos.
- Events: 74th Hunger Games, 75th Hunger Games, Quarter Quell, the Reaping,
  Dark Days, First Rebellion, Second Rebellion, bombing of District 12.
- Creatures: mockingjay, jabberjay, tracker jacker, muttation, wolf mutts.
- Systems: Hunger Games, tesserae, sponsorship, victors, Avox, district
  industry, Capitol propaganda.
- Works/adaptations: The Hunger Games trilogy, The Ballad of Songbirds and
  Snakes, Sunrise on the Reaping, film adaptations.

Allowed category concepts: characters/tributes/victors, districts/locations,
Capitol/rebel organizations, objects/weapons, Games/rebellions/events,
muttations/species, political/social systems, books/films/adaptations, and
factual adaptation-production biographies. Reject fan fiction, role-playing,
merchandise, image-only, list, template, and community pages.

### Middle-earth seeds

- Characters: Frodo Baggins, Samwise Gamgee, Gandalf, Aragorn, Legolas, Gimli,
  Boromir, Gollum, Sauron, Saruman, Galadriel, Elrond, Bilbo Baggins, Thorin,
  Fëanor, Morgoth, Beren, Lúthien, Túrin, Eärendil.
- Places: the Shire, Mordor, Gondor, Rohan, Rivendell, Lothlórien, Moria,
  Minas Tirith, Númenor, Beleriand, Valinor, Middle-earth, Erebor, Isengard.
- Organizations: Fellowship of the Ring, White Council, Istari, Dúnedain,
  Rohirrim, Rangers of the North, House of Finwë.
- Objects: One Ring, Silmarils, palantíri, Sting, Andúril, Ring of Barahir,
  Arkenstone, Phial of Galadriel.
- Events: War of the Ring, Battle of the Pelennor Fields, Fall of Númenor,
  War of the Last Alliance, Kinslaying at Alqualondë, War of Wrath, Battle of
  Five Armies.
- Creatures/peoples: Elves, Dwarves, Hobbits, Orcs, Ents, Dragons, Great
  Eagles, Balrogs.
- Systems: Ainulindalë, Music of the Ainur, Rings of Power, Tengwar, Quenya,
  Sindarin, Ages of Arda.
- Works/adaptations: The Hobbit, The Lord of the Rings, The Silmarillion,
  Unfinished Tales, Peter Jackson's film trilogies, major video games.

Allowed category concepts: legendarium characters/peoples, realms/regions/
settlements, houses/councils/fellowships, artifacts/weapons/rings, battles/
wars/ages, creatures/races, languages/writing/motifs, writings/films/games,
and factual publication/adaptation history. Reject fanon, role-playing,
merchandise, image-only, list, template, and community pages.

## Frozen source manifests (2026-08-25)

All source prose was selected and frozen before the first LLM call. Manifests
live only under the task scratch root; their digests and review evidence are
durable here.

| Universe | Candidates | Selected | Reserve | Page cap | Selected SHA-256 | Reserve SHA-256 |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| Harry Potter | 8,393 | 1,600 | 160 | 4 | `21dc168c61d3b76b4360c58873b4501d007f753757cba148b45f3fd3592594db` | `4dde91ed888edea341610f88915de1e61c351daa50a18e7afab865cd725d8125` |
| The Hunger Games | 2,125 | 1,300 | 130 | 15 | `e1f9a5f212fbb61ccda51b0b7968764b69c5b1067ef75167604f7af951b70251` | `7da77e471fa6184c8ae8d1d272614b0fd2b5e69b8c6f0a4ef55426f3dd69c23c` |
| Middle-earth | 4,226 | 1,808 | 181 | 4 | `22b400957238644f1202ba9f184a49e34853be4be1c19d68b2a6d08d5f716dea` | `ae07ed60a619b05c82c07397b1bb6facfc2b88f988680035c37daabbce4aa038` |

The selected manifests split into 18 immutable source batches: Harry Potter
`300/300/300/300/300/100`, Hunger Games `300/300/300/300/100`, and
Middle-earth `300/300/300/300/300/300/8`. Concatenating each universe's batch
files reproduces its selected manifest byte for byte.

The pre-generation review covered 30 deterministic stratified rows from every
full batch and all eight rows from `me-007`: 518/518 accepted, zero license
keys, and coverage of every present entity class, lead/non-lead sections, and
page diversity. An exhaustive second validator checked all 4,708 rows for
identity, host, revision, lengths, sentence completeness, duplicates against
the baseline and each other, continuity, class/page caps, raw markup,
extraction artifacts, low-value paths, and absent license fields.

Selector iterations found and fixed several general defects rather than
blacklisting individual pages:

- void tags inside Fandom infoboxes had trapped the HTML parser's skip depth;
- provenance and maintenance categories had distorted entity classes;
- the phrase “Hunger Games” had falsely implied video-game continuity;
- sentence splitting broke quoted exclamations, age abbreviations, initials,
  and honorifics such as `Dr.` and `Mrs.`;
- lyrics, quiz questions, publisher blurbs, DLC support text, merchandise,
  reference errors, unfinished maintenance pages, raw wiki markup, glued
  sentences, and corrupted Moon-letter spans needed explicit rejection;
- Hunger Games needed page cap 15 only to fill its quality-gated 10% reserve;
  cap 13 yielded 1,417 rows and cap 14 yielded 1,429 for a 1,430-row target.

## Continuity tags

- Harry Potter: `books`, `films`, `games`, `unspecified`
- Hunger Games: `books`, `films`, `games`, `unspecified`
- Middle-earth: `legendarium`, `films`, `games`, `unspecified`

A row carries multiple tags only when its prose explicitly combines or
compares those continuities. Fanon is always excluded.

## Prompt v1 — generation

System:

```text
You are a meticulous bilingual retrieval-dataset author and franchise terminology editor. Work only from the supplied frozen passage. Produce grounded standalone questions and canonical retrieval keyphrases. Never add outside lore. Return only the required strict JSON object.
```

User template:

```text
Create retrieval data for the frozen passage below.

Return exactly this schema:
{
  "queries_en": [three English questions],
  "queries_ru": [three Russian questions],
  "keywords_en": [four to eight English keywords or keyphrases],
  "keywords_ru": [four to eight Russian keywords or keyphrases]
}

Question intents in each language:
1. a concrete fact, entity, attribute, time, or place;
2. a relationship, cause, consequence, contrast, or role;
3. a specific standalone contextual paraphrase.

Requirements:
- Every item must be supported solely by the passage. Do not add outside lore.
- Questions must be natural, standalone, and 3–24 words long.
- Do not embed the answer or create a false premise.
- Russian must use established official names, terms, spelling, and declension.
- Never mention the passage, text, prompt, or instructions.
- No greetings, hashtags, emoji, fandom chatter, mixed-script words, or explanations.
- Keywords must be specific retrieval anchors: named entities, canonical concepts, places, organizations, events, relationships, or distinctive multi-word phrases.
- Reject generic keywords such as character, story, magic, battle, person, place, книга, персонаж, история, магия, битва, or место unless part of a specific canonical phrase.
- Align English and Russian keyword meaning naturally; do not force literal translation of official names.

Universe: {universe}
Title: {title}
Section: {section_path}
Continuity: {continuity_json}

Frozen passage:
{content}
```

Structured Outputs require exactly three query items and four to eight keyword
items per language, with `additionalProperties=false`.

## Prompt v1 — independent audit

System:

```text
You are an independent bilingual retrieval-dataset auditor and franchise terminology fact checker. Judge only against the supplied frozen passage. Return the strict audit JSON and do not silently rewrite any field.
```

User template:

```text
Audit the generated retrieval payload against the frozen passage.

Fail for any real defect:
- a question or keyword is not supported solely by the passage;
- a false premise or answer leakage;
- duplicated intent within one language;
- vague or non-standalone retrieval wording;
- unnatural grammar, bad official terminology, spelling, declension, or transliteration;
- mixed scripts, meta wording, chatty language, or generic keyword noise;
- English/Russian keywords are materially misaligned.

Return:
{
  "pass": true or false,
  "defects": [
    {"field": "queries_ru[1]", "code": "terminology", "explanation": "Specific defect grounded in the passage."}
  ]
}

Universe: {universe}
Title: {title}
Section: {section_path}
Continuity: {continuity_json}
Frozen passage:
{content}

Generated payload:
{generated_json}
```

Audit uses `gpt-5.6-sol` with high reasoning. Repair receives the audit defects
and returns the complete generation schema, also with high reasoning. Three
failed repairs reject the row; a reserve passage replaces it through a new
logged lineage.

## Terminology notes

### Harry Potter

Build the accepted Russian terminology list from repeated model audit findings
and direct source evidence. Preserve canonical English spellings in English
questions and official Russian localization in Russian questions. Do not mix
Latin and Cyrillic inside a word.

Accepted `hp-001` forms worth carrying forward include `станция Хогсмид`,
`вокзал Кингс-Кросс`, `больница Святого Мунго`, `окклюменция`, and
`Заклятие вечной клейкости`. Treat these as retrieval terminology, not as a
license to add facts absent from a frozen passage.

When a Russian localization of a proper name is uncertain, keep the canonical
English spelling as a complete token. Do not guess between competing translated
names. The rejected Babbitty Rabbitty row demonstrated why: independent high
audits contradicted each other between `Зайка Шутиха` and `Зайчиха Шутиха`.

### The Hunger Games

Keep book and film continuity separate where source prose distinguishes them.
Record official Russian names, districts, offices, events, and institutions as
they are encountered and audited.

The first `hg-001` audit established that numbered districts use the
unhyphenated form `Дистрикт 2`, and that Justice Building is `Дворец
правосудия`. Questions about a soundtrack or album must name the exact work;
`саундтрек` or `альбом` alone is not standalone. Preserve who said something
to whom, simultaneity versus causality, and epistemic qualifiers instead of
turning them into stronger claims.

### Middle-earth

Preserve Tolkien diacritics and distinguish legendarium prose from film/game
adaptations. Record established Russian forms only after they appear in an
accepted, audited batch.

Accepted `me-001` retrieval forms include `Бессмертные Земли`, `Борондир
Удалраф`, `эвкатастрофа`, and `S-руна Сарумана`. Keep canonical Latin spelling
as a complete token when the Russian localization is uncertain. Do not infer
travel method from a generic departure, convert a clue into a physical key,
or ask “why” when the frozen passage states only a relation without its cause.

## Lessons inherited from the Witcher iteration

- Source passages were reliable when extracted from rendered MediaWiki HTML
  with section hierarchy and aggressive boilerplate removal.
- One model pass was insufficient for Russian declension and official terms;
  independent editing/audit materially improved quality.
- Global duplicate validation must happen while claims are serialized, not
  only inside one record.
- Model-proposed audit replacements cannot be applied blindly; every repair
  must re-enter deterministic validation and independent audit.
- Query generation must be resumable and append output incrementally.
- Passage provenance and content must be compared before and after attachment.

### Stricter selector regression against Witcher source300

The 2026-08-25 selector regression read all 300 frozen Witcher source rows
without mutation: 300 unique chunk IDs, 300 unique contents, and maximum
pairwise 5-word-shingle Jaccard `0.06997084548104957`. The new selector would
reject 66 legacy rows: 37 below the new 420-character/70-word floor, 21 with a
category containing a fanon signal, five unresolved-pronoun openings, and
three incomplete endings. These are intentional stricter gates for the new
universes; the already accepted Witcher rows remain unchanged.

## Decision log

### 2026-08-25

- Target confirmed as exactly 10,000 passages, not 10,000 questions.
- Selected Harry Potter, Hunger Games, and Middle-earth as the next universes.
- Fixed quotas at 1,600 / 1,300 / 1,808.
- Existing legacy book excerpts remain unchanged during corpus expansion.
- Source selection precedes all LLM calls.
- Pinned `gpt-5.6-sol`; mini and GPT-4.x models are forbidden.
- GPT-5.6 generates both bilingual questions and bilingual keyphrases.
- Work proceeds in an isolated worktree and untracked scratch directory.
- QA Universe Benchmark materialization begins only after final 10k audit.
- Scratch generation/merge suite reached 43 passing tests after an expected
  missing-module RED. The exact-model probe row
  `002800c8-b3b4-5528-8d95-98492e76a0c3` returned `gpt-5.6-sol` for both
  medium generation and independent high audit, passed without repair, emitted
  3 EN + 3 RU questions and 7 EN + 7 RU keyphrases, and preserved the frozen
  passage hash byte for byte.

### Harry Potter batch `hp-001` — kept 2026-08-25

- Original frozen manifest: 300 rows, SHA-256
  `fcaddb9f2b7922d3b7b128ddfcb432099d5f07e6434e08a3221be6d9b3b2087d`.
- Medium generation returned exact `gpt-5.6-sol` for 300/300 rows. Independent
  high audit produced 512 decisions: 295 rows passed within three repairs and
  five were rejected instead of being forced through.
- The five rejects were replaced from the already frozen reserve with explicit
  rejected-to-replacement lineage and the universe-wide page cap of four.
  Replacement generation returned exact `gpt-5.6-sol` for 5/5; high audit
  accepted all five, with one row requiring one repair.
- Effective 300-row source manifest SHA-256:
  `0a28c6e4400bead26a5a5230921bc52a8cd12c677c6041bb5df85663fa219390`.
  Original source manifests and SHAs were not overwritten.
- Whole-corpus uniqueness found seven duplicate query occurrences across three
  rows. All three rows went through high repair and a fresh independent high
  audit; the final batch has zero duplicates against the 15,624-query baseline
  or within itself.
- Final repair distribution, including global repairs: 149 rows with zero
  repairs, 111 with one, 29 with two, and 11 with three. In total 151/300 final
  rows were materially repaired through 202 repair attempts.
- Canonical corpus after atomic merge: 5,592 rows, 17,424 unique normalized
  queries, 300 `harry_potter` rows, zero duplicate IDs/content, and zero
  `license` or `license_url` keys. Dataset SHA-256:
  `c7357cec8568cf7d63adf2b6fd8e53eb8f5f6ce55ea355571283b96f881eda9d`.
- Full project verification: 690 passed, one skipped. Scratch pipeline:
  48 passed.
- Private Hugging Face commit:
  `17bc9bd6dfb05170bf2967e4c287171b1bcb2876`. A clean remote download matched
  the local SHA, row count, query count, and zero-license check exactly.

Prompt lessons carried into replacements and `hp-002+`:

- State the deterministic keyphrase limit explicitly: 1–12 words and at most
  120 characters. Hidden validation constraints caused avoidable repair loops.
- Permit a canonical English proper name as a complete token inside Russian
  retrieval text when official localization is genuinely uncertain; prohibit
  mixed Cyrillic/Latin inside a single word.
- Do not weaken the three-repair limit. Difficult rows about Babbitty Rabbitty,
  Animagus ingredients, Wright & Teague/Oxfam, merpeople, and the translation
  of “most-played” were correctly discarded and replaced.

### Harry Potter batch `hp-002` — kept 2026-08-25

- Original frozen manifest: 300 rows, SHA-256
  `e579ac1bb185ed4c694386dcacdeecf1d425cb1907a20da174c2df5ad99fe03e`.
- Medium generation returned exact `gpt-5.6-sol` for 300/300 rows. Independent
  high audit produced 439 decisions: 297 rows passed within three repairs and
  three persistent failures were rejected.
- Three new frozen-reserve rows were selected after excluding the five reserve
  IDs consumed by `hp-001`. The effective 1,600-row HP selection retained its
  page cap of four. Replacement generation and high audit accepted 3/3, with
  one replacement requiring one repair.
- Effective 300-row source manifest SHA-256:
  `9072222648c9fcc3ec6e84df4b88f7a232ec783c80de934fe6308c97ba638218`.
- Whole-corpus uniqueness found two queries already present in the published
  17,424-query corpus. Both rows passed high repair and a fresh high audit;
  final global rejects were zero.
- Final repair distribution, including global repairs: 190 rows with zero
  repairs, 90 with one, 17 with two, and three with three. In total 110/300
  final rows were materially repaired through 133 repair attempts.
- Canonical corpus after atomic merge: 5,892 rows, 19,224 unique normalized
  queries, 600 `harry_potter` rows, and zero license keys. Dataset SHA-256:
  `cb1ea5a2de4cb502dbf98f6badb93cb7becccb26996483ed120147e4e030cc04`.
- Prompt memory measurably reduced audit load: zero-repair rows increased from
  146 in `hp-001` to 190, audit decisions fell from 512 to 439, and terminology
  defects fell from 101 to 52.
- Persistent rejects covered the Improper Use of Magic Office, the modelling
  term `special bookings`, and an ambiguous Legilimency sentence about Jacob
  and a Cursed Vault. All three were replaced rather than weakened through.
- Validation now recognises Unicode-capitalised canonical names such as
  `École D'Apparition Démoniste Limus` and English titles with lowercase
  connectors, while still rejecting lowercase untranslated generic phrases.
- Full project verification: 690 passed, one skipped. Scratch pipeline:
  48 passed.
- Private Hugging Face commit:
  `a113dfa5670ce192ec9dfd26f5db6fa3d2ba2ac8`. A clean remote download matched
  the local SHA, row count, query count, and zero-license check exactly.

### Harry Potter batch `hp-003` — kept 2026-08-27

- Original frozen manifest: 300 rows, SHA-256
  `c3cb53f5f248408e327078f4b33687ceca41694eae2380a0da6b4df8dda964a0`.
- Medium generation returned exact `gpt-5.6-sol` for 300/300 rows. Independent
  high audit produced 449 original decisions: 299 rows passed within three
  repairs and one persistent failure was rejected.
- The rejected Three Broomsticks row repeatedly overstated the source's phrase
  “the second time the pub had been hit”. It was replaced rather than weakened
  through. One new row from the frozen reserve passed generation and high audit
  on its first attempt, with all eight earlier reserve IDs excluded.
- Effective 300-row source manifest SHA-256:
  `bb38f86f2d10cf7c03f583b867bbae1295abc3ee41634faacfeea10a56cf19bd`.
  The six-batch Harry Potter selection still has at most four chunks per page.
- Whole-corpus uniqueness found the English and Russian forms of a Neville
  Longbottom birth-date question already present in the 19,224-query corpus.
  One high repair plus a fresh independent high audit removed both collisions;
  final global rejects were zero.
- Final repair distribution: 183 rows with zero repairs, 93 with one, 18 with
  two, and six with three. In total 117/300 final rows were materially repaired
  through 147 final repair attempts.
- Canonical corpus after atomic merge: 6,192 rows, 21,024 unique normalized
  queries, 900 `harry_potter` rows, zero duplicate IDs/content, and zero
  `license` or `license_url` keys. Dataset SHA-256:
  `911a9843a774af8e6fb0e3715a8cb010c4ed3e646d21c8de05b27f82a9e65489`.
- A paid-call usage ledger was enabled before resuming the batch. The final
  repair and audit used 2,041 input tokens and 1,044 output tokens, including
  749 reasoning tokens, for an estimated `$0.029044` at the current
  `gpt-5.6-sol` rates. Reasoning tokens are part of output and are not charged
  twice.
- Full project verification: 690 passed, one skipped. Scratch pipeline:
  50 passed. The first 5,892 corpus rows remained unchanged.
- Private Hugging Face commit:
  `7af7f79f69e75dd69912f867736189bf09b02778`. A clean remote download matched
  the local 6,192-row file and SHA exactly. Dataset Viewer reported no pending
  or failed jobs; the downloaded server-side Parquet contained 6,192 rows and
  30 columns.

### Harry Potter batch `hp-004` — kept 2026-08-28

- Original frozen manifest: 300 rows, SHA-256
  `72f9874c98d9d34424f2376d110f0f741409d1a557cb99d264441a575ba2bed3`.
  Medium generation returned exact `gpt-5.6-sol` for 300/300 rows. Six
  responses were repeated only after their first structured payload could not
  be parsed; source order and IDs remained exact.
- Independent high audit produced 457 original audit events. It accepted 299
  rows within the three-repair limit and rejected one persistent failure. The
  rejected Ministry visitor-entrance row repeatedly rendered the `62442`
  mnemonic as a bare Latin-script common word in a Russian keyphrase, or
  changed the ownership of the red telephone box. It was replaced rather than
  weakening the audit.
- One new frozen-reserve row was selected after excluding all reserve IDs used
  by `hp-001` through `hp-003` and the three already assigned to `hp-005`.
  Generation and independent high audit accepted the replacement on their
  first attempts. Effective 300-row source SHA-256:
  `fa8823daac5c759b892746ea39c98f14806ece53fffa989c4c0e5f61af2c82fc`.
  The complete 1,600-row Harry Potter selection still respects the four-chunk
  page cap.
- Whole-corpus uniqueness found three conflicting rows against the 21,024-query
  baseline. All three passed one high repair and a fresh high audit; final
  global rejects were zero. Final repair distribution: 176 rows with zero
  repairs, 98 with one, 19 with two, and seven with three. In total 124/300
  final rows were materially repaired through 157 repair attempts.
- The paid-call ledger contains 923 hp-004 calls: 307 generation calls, 453
  batch audits, 157 batch repairs, and six synchronous global-finalization
  calls. They used 858,569 input and 872,035 output tokens (719,224 reasoning
  tokens included in output) for an estimated `$10.589180`. Batch generation,
  audit, and repair used the official 0.5 price multiplier; response IDs are
  unique and crash-safe in the ledger.
- Canonical corpus after atomic merge: 6,492 rows, 22,824 unique normalized
  queries, 1,200 `harry_potter` rows, zero duplicate IDs/content, and zero
  `license` or `license_url` keys. Dataset SHA-256:
  `a21db172b5bd3e14505be36b17f9c5ad1cfb73643546114a7518598d0001db56`.
- Full project verification: 690 passed, one skipped. Scratch pipeline:
  58 passed.
- Private Hugging Face commit:
  `6a0bc1e8f3ee2bf1949561b3019faf391402718b`. A clean remote download matched
  the local 6,492-row file byte-for-byte. Dataset Viewer reported
  `pending=[]`, `failed=[]`, and `partial=false`; its server-side Parquet
  contained 6,492 rows, 30 columns, 22,824 queries, 1,200 Harry Potter rows,
  and no license columns.
- Prompt memory for later HP batches: preserve the passage's grammatical
  ownership for physical entrances; do not place a bare lowercase Latin
  common word into a Russian keyphrase merely because the source gives an
  English mnemonic in parentheses. Use a Cyrillic explanation such as the
  number's relation to the word only when the passage supports it.

### Harry Potter batch `hp-005` — kept 2026-08-28

- Original frozen manifest: 300 rows, SHA-256
  `d2d713957a5e2207f07409d594a103e2a67584e1601aba712f796047c965809b`.
  Medium Batch-API generation produced exact `gpt-5.6-sol` payloads for all
  rows; five responses with unparseable first payloads were repeated
  individually, never as a full-batch retry.
- Independent high audit produced 461 original audit events. It accepted 297
  rows within three repairs and persistently rejected three: a Sphinx passage
  with unstable Beast/Being Russian taxonomy, an Unforgivable History passage
  whose Russian wording lost the Triwizard golden-egg Foundable anchor, and a
  Hogwarts Library passage that repeatedly invented ownership or gender.
- Three unused frozen-reserve rows replaced those rejects and passed generation
  plus high audit on their first attempts. Effective 300-row source SHA-256:
  `e301a3b89aaa7230d289ee34ffc36350ba9590eb6e10c7d2f00e47dc0d5c4f58`.
  Reserve lineage remains disjoint across `hp-001` through `hp-005`, and the
  complete HP selection retains its four-chunk page cap.
- Initial whole-corpus finalization repaired seven colliding rows against the
  6,192-row baseline. Because `hp-004` was committed first, all 300 already
  repaired payloads were deterministically rechecked against the new 6,492-row
  baseline; exactly one new hp-004 collision required one high repair and a
  fresh high audit. Final global rejects were zero.
- Final repair distribution: 183 rows with zero repairs, 83 with one, 25 with
  two, eight with three, and one cumulative six-repair row after both global
  finalization passes. In total 117/300 final rows were materially repaired;
  the stored cumulative repair count is 163.
- The paid-call ledger contains 953 hp-005 calls and unique response IDs. They
  used 888,298 input and 920,295 output tokens (759,128 reasoning tokens
  included in output) for an estimated `$11.314200`. Of these, 931 calls used
  the official Batch 0.5 multiplier; 22 sequential global-finalization calls
  were synchronous because each row had to reserve its questions before the
  next row was checked.
- Canonical corpus after atomic merge: 6,792 rows, 24,624 unique normalized
  queries, 1,500 `harry_potter` rows, zero duplicate IDs/content, and zero
  `license` or `license_url` keys. Dataset SHA-256:
  `50c61a7471210e269cb481f6490021711cc477eee3685344d4ab40314e7ee948`.
- Full project verification: 690 passed, one skipped. Scratch pipeline:
  58 passed.
- Private Hugging Face commit:
  `68c632f20b7d0dd0df3ff3209acee844104505c4`. A clean remote download matched
  the local file byte-for-byte. Dataset Viewer reported `pending=[]`,
  `failed=[]`, and `partial=false`; its server-side Parquet contained 6,792
  rows, 30 columns, 24,624 queries, 1,500 Harry Potter rows, and no license
  columns.
- Prompt memory for later rows: never infer a creature belongs to the person
  who merely transported it; preserve gender-neutral source wording for Rowan
  Khanna; avoid forcing uncertain franchise common-noun taxonomy into a mixed
  Latin/Cyrillic Russian phrase. If a term such as Foundable is not necessary
  for a grounded standalone query, anchor the question on supported concrete
  events instead of inventing a localization.

### Harry Potter batch `hp-006` and milestone — kept 2026-08-28

- Original final HP manifest: 100 rows, SHA-256
  `9a7f7586819e80488339f1fdfd0e8e225882d99721abe34288712b2f0f0a7d25`.
  Medium generation returned exact `gpt-5.6-sol` payloads for 100/100 rows;
  three first responses with unparseable payloads were repeated individually.
- Independent high audit produced 154 original events and accepted 99 rows.
  The persistent Dursley etymology reject repeatedly converted the passage's
  hedged “likely” influence into a fact, confused a surname with a family or
  given name, or used an incorrect Russian singular/plural form. It was
  replaced rather than weakened through.
- One unused frozen-reserve row was selected after excluding every reserve ID
  consumed by `hp-001` through `hp-005`. Its first two audited versions had
  Russian title-case and unsupported-surname defects; the second repair passed
  a third independent high audit. Effective 100-row source SHA-256:
  `e1ba0c0a22b55c1e7df806e9f673cbf95f8b766695fb6f9068632b7536d51ca8`.
  All six effective manifests together contain 1,600 unique rows and retain
  the four-chunk page cap.
- Whole-corpus finalization found zero query collisions against the 24,624-query
  baseline. Final repair distribution: 60 rows with zero repairs, 29 with one,
  nine with two, and two with three. In total 40/100 rows were materially
  repaired through 53 stored repair attempts; global rejects were zero.
- The paid-call ledger contains 316 hp-006 calls, all at the Batch 0.5 price
  multiplier: 104 generation, 156 audit, and 56 repair calls. They used
  291,215 input and 323,034 output tokens (269,650 reasoning tokens included
  in output) for an estimated `$3.848247`.
- Canonical corpus after atomic merge: 6,892 rows, 25,224 unique normalized
  queries, exactly 1,600 `harry_potter` rows, zero duplicate IDs/content, and
  zero `license` or `license_url` keys. Dataset SHA-256:
  `54dc7dd8351e8c7a2a02795088f0649d8a08989b3afce344352384d7c27b1263`.
- Full project verification: 690 passed, one skipped. Scratch pipeline:
  58 passed.
- Private Hugging Face commit:
  `86813e8a6c5cf029d8e61b5c03ec0dd1b2da746e`. The authenticated raw download
  matched the local file byte-for-byte. Dataset Viewer reported `pending=[]`,
  `failed=[]`, and `partial=false`; its server-side Parquet contained 6,892
  rows, 30 columns, 25,224 queries, 1,600 Harry Potter rows, and no license
  columns.
- The Harry Potter milestone is complete: 1,600 new wiki passages and 9,600
  bilingual grounded questions across six kept batches, with 14 persistent
  rows replaced from the frozen reserve. The reliable paid-call ledger covers
  `hp-004` through `hp-006` at an estimated `$25.751627`; earlier batches are
  not retroactively assigned invented token costs.
- Prompt memory carried forward: preserve epistemic qualifiers such as
  “likely”; distinguish surnames, family plurals, and given names; use Russian
  sentence-style capitalization inside translated titles; never infer that a
  sibling shares a surname when the source names only the given name.

### Middle-earth batch `me-001` — kept 2026-08-28

- Original manifest: 300 unique frozen Tolkien Gateway passages. Medium
  generation produced 300/300 strict payloads; one difficult row exhausted
  its 2,200-token output budget twice before the third exact request completed.
  Including the one later reserve row, generation produced 301 kept payloads.
- Independent high audit accepted 299 original rows after up to three bounded
  repairs and rejected one persistent row. A fresh frozen-reserve passage was
  selected under the four-chunk page cap; it passed after one repair and a new
  independent audit. Effective source SHA-256:
  `77d6ed46d554f470996f4e7386fb7244c961279bcea6551f5676d74b5bcb4820`.
- Whole-corpus finalization found zero query collisions. Final repair
  distribution: 199 rows with zero repairs, 88 with one, 11 with two, and two
  with three; global rejects were zero.
- The complete paid-call ledger for the original and reserve lifecycles has
  837 Batch calls: 303 generation, 415 audit, and 119 repair. They used 734,491
  input and 763,723 output tokens, including 633,967 reasoning tokens, for an
  estimated `$9.165153`.
- Canonical corpus after atomic merge: 7,192 rows, 27,024 unique normalized
  queries, exactly 300 `middle_earth` rows, zero duplicate IDs/content, and no
  `license` or `license_url` keys. Dataset SHA-256:
  `c3660be7b597b4ad8b536d35a40e8d8f045a0a19a3cc50e7cb9e4a2d0c4e9138`.
- Full project verification: 690 passed, one skipped. Scratch pipeline:
  58 passed. Private Hugging Face commit:
  `bda13eda07245f28142e2df3a8f1bade2503236a`; the authenticated remote raw
  download matched the local file byte-for-byte. Dataset Viewer reported
  `pending=[]`, `failed=[]`, and `partial=false`; its server-side Parquet had
  7,192 rows, 30 columns, 27,024 queries, 300 Middle-earth rows, 1,600 Harry
  Potter rows, and no license columns.
- Prompt memory carried forward: retain Tolkien diacritics and canonical Latin
  tokens; distinguish departure from its unstated travel method, a clue from a
  physical key, and a stated dependency from an unstated causal explanation.

### The Hunger Games batch `hg-001` — kept 2026-08-28

- Original manifest: 300 unique frozen Hunger Games Wiki passages, SHA-256
  `3dd843f8070881873e9de74a2ad513fb2f1a4bca272e6cf45c6199810e30e0e1`.
  Medium generation completed 300/300 strict payloads after five targeted
  retries. Including the later reserves, generation produced 303 kept
  payloads without resubmitting completed rows.
- Independent high audit produced 487 original audit events, accepted 297
  original rows after up to three bounded repairs, and rejected three. Three
  fresh frozen-reserve passages each passed after one repair and a new audit.
  Effective source SHA-256:
  `b85390b1c9a80727d1ddee98618512e1eacc86dae10093b186b8d00a2585656e`.
- Whole-corpus finalization found one query collision against the 7,192-row
  baseline, repaired it, and independently re-audited the changed payload.
  Final repair distribution: 157 rows with zero repairs, 108 with one, 31 with
  two, and four with three; global rejects were zero.
- The complete paid-call ledger contains 986 calls: 984 Batch calls and two
  synchronous global-finalization calls. Phase totals are 308 generation, 487
  audit, and 191 repair calls. They used 849,243 input and 938,666 output
  tokens, including 780,168 reasoning tokens, for an estimated `$11.156844`;
  the two full-price calls account for `$0.046724`.
- Canonical corpus after atomic merge: 7,492 rows, 28,824 unique normalized
  queries, exactly 300 `the_hunger_games` rows and 300 `middle_earth` rows,
  zero duplicate IDs/content, and no `license` or `license_url` keys. Dataset
  SHA-256:
  `1abab9748233c29febff8c8a425a2c9881c99706214a0e2bd064c01cc8521d8b`.
- Full project verification: 690 passed, one skipped. Scratch pipeline:
  58 passed. Private Hugging Face commit:
  `43a72ab21b106398c347e9125bfc7340871821a9`; the authenticated remote raw
  download matched the local file byte-for-byte. Dataset Viewer reported
  `pending=[]`, `failed=[]`, and `partial=false`; its server-side Parquet had
  7,492 rows, 30 columns, 28,824 queries, 300 Hunger Games rows, 300
  Middle-earth rows, 1,600 Harry Potter rows, and no license columns.
- Prompt memory carried forward: write numbered districts without a hyphen,
  name the exact soundtrack or album in standalone questions, and preserve
  speaker/addressee, simultaneity, causality, and epistemic qualifiers.

### Middle-earth batch `me-002` — kept 2026-08-28

- Original manifest: 300 unique frozen Tolkien Gateway passages, SHA-256
  `95a4841b85dc493aa2ad881637a27eceec8a5290b725f43e1ca512bc2b5e9971`.
  Medium generation completed 300/300 payloads after one targeted retry.
- Independent high audit accepted 296 original rows after bounded repairs and
  rejected four. Four unused frozen-reserve passages were selected after
  excluding the reserve consumed by `me-001`; all four passed the full
  generation, repair, and re-audit gates. Effective source SHA-256:
  `9d14661747fa84113c0b01c26400a3506b61f20d894783f2d2a0e86e19ee7fad`.
- Whole-corpus finalization found collisions in two rows. Three total global
  repair attempts produced independently audited replacements with no final
  rejects. Final repair distribution: 190 rows with zero repairs, 93 with one,
  16 with two, and one with three.
- The complete paid-call ledger contains 877 calls: 871 Batch and six
  synchronous global-finalization calls. Phase totals are 305 generation, 432
  audit, and 140 repair calls. They used 763,649 input and 773,985 output
  tokens, including 635,519 reasoning tokens, for an estimated `$9.434712`;
  the six full-price calls account for `$0.205372`.
- Canonical corpus after atomic merge: 7,792 rows, 30,624 unique normalized
  queries, exactly 600 `middle_earth` rows and 300 `the_hunger_games` rows,
  zero duplicate IDs/content, and no `license` or `license_url` keys. Dataset
  SHA-256:
  `433f35d608d154789182aac49f641e2a9030d1a01f74718385ae7b4d1048ee79`.
- Full project verification: 690 passed, one skipped. Scratch pipeline:
  58 passed. Private Hugging Face commit:
  `00ca9db762b141a1edf306ab68b4d432e1dc79f3`; the authenticated remote raw
  download matched the local file byte-for-byte. Dataset Viewer reported
  `pending=[]`, `failed=[]`, and `partial=false`; its server-side Parquet had
  7,792 rows, 30 columns, 30,624 queries, 600 Middle-earth rows, 300 Hunger
  Games rows, 1,600 Harry Potter rows, and no license columns.

### Middle-earth batch `me-003` — kept 2026-08-28

- Original manifest: 300 unique frozen Tolkien Gateway passages, SHA-256
  `adf9264308d02fce4ce741c47423f49c34322d1f887ee02699a9010b2ee0f94b`.
  Medium generation completed 300/300 strict payloads after one targeted
  retry.
- Independent high audit accepted 299 original rows after bounded repairs and
  rejected one. One unused frozen-reserve passage passed generation and its
  first independent audit. Effective source SHA-256:
  `78ecbab1b034d78360b72e9303745b63dd56cce1aafdebab8df9907779727ef7`.
- Whole-corpus finalization found and repaired one global query collision, then
  independently re-audited the changed payload. Final repair distribution:
  202 rows with zero repairs, 80 with one, 16 with two, and two with three;
  global rejects were zero.
- The complete paid-call ledger contains 836 calls: 834 Batch and two
  synchronous global-finalization calls. Phase totals are 302 generation, 413
  audit, and 121 repair calls. They used 725,239 input and 773,015 output
  tokens, including 641,350 reasoning tokens, for an estimated `$9.256925`;
  the two full-price calls account for `$0.031848`.
- Canonical corpus after atomic merge: 8,092 rows, 32,424 unique normalized
  queries, exactly 900 `middle_earth` rows and 300 `the_hunger_games` rows,
  zero duplicate IDs/content, and no `license` or `license_url` keys. Dataset
  SHA-256:
  `cdaa6a765e1a39469f6c2a1c4f645c7632ca187313d8d4ce1e8b829575220ec6`.
- Full project verification: 690 passed, one skipped. Scratch pipeline:
  58 passed. Private Hugging Face commit:
  `80ca9b408958b6de2241e5b9c0c019a617140406`; the authenticated remote raw
  download matched the local file byte-for-byte. Dataset Viewer reported
  `pending=[]`, `failed=[]`, and `partial=false`; its server-side Parquet had
  8,092 rows, 30 columns, 32,424 queries, 900 Middle-earth rows, 300 Hunger
  Games rows, 1,600 Harry Potter rows, and no license columns.

### The Hunger Games batch `hg-002` — kept 2026-08-28

- Original manifest: 300 unique frozen Hunger Games Wiki passages, SHA-256
  `4d719c817dcbba3d8112adb46033d6078fcb1097edd1ecb191ec8d5b65bcbacc`.
  Medium generation completed 300/300 strict payloads after three targeted
  retries without resubmitting successful rows.
- Independent high audit produced 481 original audit events, accepted 299
  original rows after bounded repairs, and rejected one. One unused
  frozen-reserve passage then passed generation and its first independent
  audit. Effective source SHA-256:
  `f81fc678ce3636705795ce44c6ad07bb9961093f0b4280f7b1e5489203ca016b`.
- All eight accepted originals that required the maximum three repairs were
  manually checked against their frozen passages and retained. The review
  confirmed that uncertain legal restrictions remained explicitly unknown,
  weaker implications were not promoted to facts, and compound consequences
  stayed answerable from the source.
- Whole-corpus finalization found five rows with query collisions against the
  8,092-row baseline. Each was repaired once and independently re-audited;
  final rejects were zero. Final repair distribution: 164 rows with zero
  repairs, 97 with one, 31 with two, and eight with three.
- The complete paid-call ledger contains 968 calls: 958 Batch and ten
  synchronous global-finalization calls. Phase totals are 304 generation, 478
  audit, and 186 repair calls. They used 832,529 input and 927,598 output
  tokens, including 774,245 reasoning tokens, for an estimated `$11.107006`;
  the ten full-price calls account for `$0.220976`.
- Canonical corpus after atomic merge: 8,392 rows, 34,224 unique normalized
  queries, exactly 600 `the_hunger_games` rows and 900 `middle_earth` rows,
  zero duplicate IDs/content, and no `license` or `license_url` keys. Dataset
  SHA-256:
  `3e619b8d746123cba7ad76dc0dd1c7c83d1a9753b476394fa91d11d0737c9ce1`.
- Full project verification: 690 passed, one skipped. Scratch pipeline:
  58 passed. Private Hugging Face commit:
  `577e1f14d53de4deb1d36097f50249b5067823cc`; the authenticated remote raw
  download matched the local file byte-for-byte. Dataset Viewer reported
  `pending=[]`, `failed=[]`, and `partial=false`; its server-side Parquet had
  8,392 rows, 30 columns, 34,224 queries, 600 Hunger Games rows, 900
  Middle-earth rows, 1,600 Harry Potter rows, and no license columns.
- Prompt memory carried forward: preserve the distinction between an unknown
  rule and an implied tendency; when a condition has multiple explicit
  consequences, do not collapse them into a stronger single claim.

### Middle-earth batch `me-004` — kept 2026-08-28

- Original manifest: 300 unique frozen Tolkien Gateway passages, SHA-256
  `b5c6f84725c1f675fdc50b89bd97da1108edf8732ada61cbea48c96e359e0bb4`.
  Medium generation completed 300/300 strict payloads after three targeted
  retries without resubmitting successful rows.
- Independent high audit produced 435 original audit events, accepted 298
  original rows after bounded repairs, and rejected two. Two unused
  frozen-reserve passages passed the full replacement gate: one on its first
  audit and one after a single repair. Effective source SHA-256:
  `4e780c5535b3fbe9819eec7fd42c8b67dff350d25dd63b9777b817bf91a51967`.
- Whole-corpus finalization found one row with a query collision against the
  8,392-row baseline. It was repaired and independently re-audited; final
  rejects were zero. Final repair distribution: 191 rows with zero repairs,
  89 with one, 18 with two, and two with three.
- The complete paid-call ledger contains 871 calls: 869 Batch and two
  synchronous global-finalization calls. Phase totals are 305 generation, 429
  audit, and 137 repair calls. They used 767,379 input and 781,399 output
  tokens, including 644,642 reasoning tokens, for an estimated `$9.441902`;
  the two full-price calls account for `$0.038374`.
- Canonical corpus after atomic merge: 8,692 rows, 36,024 unique normalized
  queries, exactly 1,200 `middle_earth` rows and 600 `the_hunger_games` rows,
  zero duplicate IDs/content, and no `license` or `license_url` keys. Dataset
  SHA-256:
  `9a8ebf7a4cf57df03ad96697f303329b029a08789a723812b332edbf0a85cd9c`.
- Full project verification: 690 passed, one skipped. Scratch pipeline:
  58 passed. Private Hugging Face commit:
  `6bb65e6c687cb67cde8dcb06d986a1f8e209b303`; the authenticated remote raw
  download matched the local file byte-for-byte. Dataset Viewer reported
  `pending=[]`, `failed=[]`, and `partial=false`; its server-side Parquet had
  8,692 rows, 30 columns, 36,024 queries, 1,200 Middle-earth rows, 600 Hunger
  Games rows, 1,600 Harry Potter rows, and no license columns.
- Prompt memory carried forward: preserve Tolkien diacritics such as `Roäc`;
  scope facts to the named draft or latest version, and do not detach a stated
  reason such as the cessation of vigilance from that version context.

### The Hunger Games batch `hg-005` — kept 2026-08-28

- Original manifest: 100 unique frozen Hunger Games Wiki passages, SHA-256
  `c4c3375f72629e20fba2a50a9749a2c943b527d374b03c9dad54d0a90781c960`.
  Medium generation completed 100/100 strict payloads without a retry.
- Independent high audit produced 168 original audit events, accepted 98
  original rows after bounded repairs, and rejected two. Two unused
  frozen-reserve passages each passed after one repair and a new independent
  audit. Effective source SHA-256:
  `cf0946302d735c228f14a7b25a5e06f3933f1e91f93e59a7bb6e54dddaaae15b`.
- All three accepted rows that required the maximum three repairs were
  manually checked against their frozen passages and retained. Film-only
  details, the reason for Peeta's gesture, and the effects of Peeta's earlier
  accident and artificial leg remained answerable without added facts.
- Whole-corpus finalization found one row with a query collision against the
  8,692-row baseline. It was repaired and independently re-audited; final
  rejects were zero. Final repair distribution: 51 rows with zero repairs, 36
  with one, ten with two, and three with three.
- The complete paid-call ledger contains 345 calls: 343 Batch and two
  synchronous global-finalization calls. Phase totals are 102 generation, 172
  audit, and 71 repair calls. They used 295,198 input and 331,413 output
  tokens, including 276,071 reasoning tokens, for an estimated `$3.940145`;
  the two full-price calls account for `$0.029408`.
- Canonical corpus after atomic merge: 8,792 rows, 36,624 unique normalized
  queries, exactly 700 `the_hunger_games` rows and 1,200 `middle_earth` rows,
  zero duplicate IDs/content, and no `license` or `license_url` keys. Dataset
  SHA-256:
  `eb4c038aad8eeb34bb2a64edfa67350986ac1e5f15412ef38fb79e7d3142a318`.
- Full project verification: 690 passed, one skipped. Scratch pipeline:
  58 passed. Private Hugging Face commit:
  `c63ab906f26bbd23de38e1b3f8a9ce3039df9bcb`; the authenticated remote raw
  download matched the local file byte-for-byte. Dataset Viewer reported
  `pending=[]`, `failed=[]`, and `partial=false`; its server-side Parquet had
  8,792 rows, 30 columns, 36,624 queries, 700 Hunger Games rows, 1,200
  Middle-earth rows, 1,600 Harry Potter rows, and no license columns.
- Prompt memory carried forward: keep film-only details scoped to the film;
  preserve modal wording such as `could`, `likely`, and `presumably` when the
  source describes a possible post-Games role rather than a confirmed fact.

### Middle-earth batch `me-005` — kept 2026-08-28

- Original and effective manifest: 300 unique frozen Tolkien Gateway passages,
  SHA-256
  `5d6211eb95395c1fef705bb87f7aae8d6e7d03647c1cf5e69ef7404ebdd2e83e`.
  Medium generation completed 300/300 strict payloads after three targeted
  retries without resubmitting successful rows.
- Independent high audit produced 439 events and accepted all 300 originals
  after bounded repairs; no reserve replacement was needed.
- Whole-corpus finalization found two rows with query collisions against the
  8,792-row baseline. One passed after a single repair; the other's first
  repair was rejected because Russian `ревность` incorrectly suggested
  romantic or possessive jealousy instead of the rivalry between Fëanor and
  Fingolfin. Its second repair used `соперничество` and passed. Final rejects
  were zero; the two rows required three global repairs in total.
- All eight final rows with at least three repairs were manually checked
  against their frozen passages and retained. Version/adaptation scope,
  epistemic modals, named leaders, and stated causal links remained intact.
  Final repair distribution: 193 rows with zero repairs, 81 with one, 18 with
  two, seven with three, and one with four after its extra global repair.
- The complete paid-call ledger contains 876 calls: 870 Batch and six
  synchronous global-finalization calls. Phase totals are 303 generation, 431
  audit, and 142 repair calls. They used 775,709 input and 810,238 output
  tokens, including 670,009 reasoning tokens, for an estimated `$9.810118`;
  the six full-price calls account for `$0.164047`.
- Canonical corpus after atomic merge: 9,092 rows, 38,424 unique normalized
  queries, exactly 1,500 `middle_earth` rows and 700 `the_hunger_games` rows,
  zero duplicate IDs/content, and no `license` or `license_url` keys. Dataset
  SHA-256:
  `31b08be9cb67fd7d83b8587f5ee7f773bba3f1c186914bcfb45e4ca473d5117e`.
- Full project verification: 690 passed, one skipped. Scratch pipeline:
  58 passed. Private Hugging Face commit:
  `b81e6ca8ebcbb656fa9627aead1467f61ae7ef55`; the authenticated remote raw
  download matched the local file byte-for-byte. Dataset Viewer reported
  `pending=[]`, `failed=[]`, and `partial=false`; its server-side Parquet had
  9,092 rows, 30 columns, 38,424 queries, 1,500 Middle-earth rows, 700 Hunger
  Games rows, 1,600 Harry Potter rows, and no license columns.
- Prompt memory carried forward: translate contextual `jealousies` as rivalry
  or mutual suspicion when the source describes political/familial conflict;
  keep `may`, `unlikely`, and adaptation or draft scope explicit.

### Middle-earth batch `me-007` — kept 2026-08-28

- Original and effective manifest: eight unique frozen Tolkien Gateway
  passages, SHA-256
  `afab79effe705538fcff4dcce9bfeae7883048267867ae026501ab74ff564052`.
  Medium generation completed 8/8 strict payloads without a retry.
- Independent high audit accepted all eight after five bounded repairs and 13
  audit calls; no reserve replacement was needed. Every final row was then
  manually checked against its complete frozen passage, not sampled.
- The manual review verified all six questions per row, including Goldberry's
  ownership and fear qualifiers, Aragorn's route and Gandalf's note, the
  uncertainty around the first proposer of `The Two Towers`, platform-specific
  game-release facts, and Arda's geographic changes. No paid rewrite was
  warranted after review.
- Whole-corpus finalization against the 9,092-row baseline found zero query
  collisions and zero rejects. Final repair distribution: four rows with zero
  repairs, three with one, and one with two.
- The complete paid-call ledger contains 26 Batch calls: eight generation, 13
  audit, and five repair calls. They used 23,403 input and 21,917 output tokens,
  including 17,702 reasoning tokens, for an estimated `$0.270216`.
- Canonical corpus after atomic merge: 9,100 rows, 38,472 unique normalized
  queries, exactly 1,508 `middle_earth` rows and 700 `the_hunger_games` rows,
  zero duplicate IDs/content, and no `license` or `license_url` keys. Dataset
  SHA-256:
  `712c953c4fd9db250169f389b192fea311f4a0da7ad2b405554a3fd22f3c7867`.
- Full project verification: 690 passed, one skipped. Scratch pipeline:
  58 passed. Private Hugging Face commit:
  `f7903c8187cb7ba085bd76a4f4e566e3fd86ef67`; the authenticated remote raw
  download matched the local file byte-for-byte. Dataset Viewer reported
  `pending=[]`, `failed=[]`, and `partial=false`; its server-side Parquet had
  9,100 rows, 30 columns, 38,472 queries, 1,508 Middle-earth rows, 700 Hunger
  Games rows, 1,600 Harry Potter rows, and no license columns.

### Hunger Games batch `hg-004` — kept 2026-08-28

- Original manifest: 300 unique frozen passages, SHA-256
  `07896b1e5f32fc1aedea9db14a7a4d81cd3547eee21db7b83299f2eddb34d0af`.
  The final effective manifest SHA is
  `e6bef23be9c291093e698ef8b70c9748e0f606ede50ca0d269b3876bb1b12fb1`.
- Independent high audit accepted 296 originals and rejected four. The first
  reserve cycle generated four unused frozen passages and retained three. A
  Burdock Everdeen reserve still contained hyphenated `Дистрикт-12` after a
  passing audit; the manual terminology gate reopened it. It exhausted three
  repairs and was rejected instead of being manually edited. A fifth reserve,
  the Gamemaker arena passage, was freshly generated and passed high audit.
  The final 300 rows therefore contain four reserve substitutions while five
  reserve candidates were consumed.
- Whole-corpus finalization against the 9,100-row baseline produced eight
  global-audit events and zero final rejects. All six final rows with three
  repairs were manually checked against their complete passages and passed.
  Final repair distribution: 169 rows with zero repairs, 96 with one, 29 with
  two, and six with three.
- The complete paid-call ledger contains 984 calls: 975 Batch and nine
  synchronous calls. Phase totals are 308 generation, 489 audit, and 187
  repair calls. They used 841,676 input and 929,868 output tokens, including
  771,797 reasoning tokens, for an estimated `$11.099275`; synchronous calls
  account for `$0.142548`.
- Canonical corpus after atomic merge: 9,400 rows, 40,272 unique queries,
  exactly 1,000 Hunger Games rows, zero exact duplicate IDs/content, and no
  `license` or `license_url` keys. Dataset SHA-256:
  `b957e4bc130a8b5136caf96ea22340b27daab3630f7a486be689648e3b5ea14e`.
- Full project verification: 690 passed, one skipped. Scratch pipeline:
  58 passed. Private Hugging Face commit:
  `236ebeda435a342848e88a0503334b03b8671c53`; the authenticated remote raw
  download matched byte-for-byte. Its pending Viewer conversion was
  superseded by the next 9,700-row upload, whose completed Parquet includes and
  independently verifies all `hg-004` rows.

### Hunger Games batch `hg-003` — kept 2026-08-28

- Original manifest: 300 unique frozen passages, SHA-256
  `7510b2fffbbea4627a3f39c21ca5645050fc04158fed6f264f2e2c608e4a511a`.
  Independent high audit accepted 297 and rejected three after bounded
  repairs. Three unused frozen reserves (Rachel Zegler, Lenore Dove Baird, and
  Mayor Lipp) were generated and all passed; effective manifest SHA is
  `ee15af778584a84463bee2dcb0ec25560a241b1a027fb4e9a62d6b0a642b066e`.
- All three reserve rows were manually checked against their full passages.
  All eight final rows with three repairs were also manually reviewed. Film
  scope, stated causal links, speculation qualifiers, named districts, and
  franchise terminology remained supported.
- Whole-corpus finalization ran against the updated 9,400-row baseline,
  produced ten global-audit events, and ended with 300 final rows and zero
  rejects. Final repair distribution: 170 rows with zero repairs, 99 with one,
  23 with two, and eight with three.
- The complete paid-call ledger contains 958 calls: 948 Batch and ten
  synchronous calls. Phase totals are 307 generation, 473 audit, and 178
  repair calls. They used 826,975 input and 930,367 output tokens, including
  773,928 reasoning tokens, for an estimated `$11.114225`; synchronous calls
  account for `$0.210280`.
- Canonical corpus after atomic merge: 9,700 rows, 42,072 unique queries,
  exactly 1,300 Hunger Games rows, 1,508 Middle-earth rows, 1,600 Harry Potter
  rows, and 300 Witcher rows. Exact IDs, content, and queries are unique; no
  license fields remain. Dataset SHA-256:
  `bc6fef694f9175f6a534a768b197234ff0eab575e9d6f8383d6e3fb891788c18`.
- Full project verification: 690 passed, one skipped. Scratch pipeline:
  58 passed. Private Hugging Face commit:
  `fda54fb8a35ea6cf03365469e5f35e34ec542c44`; the authenticated remote raw
  download matched byte-for-byte. Dataset Viewer reported `pending=[]`,
  `failed=[]`, and `partial=false`; authenticated server Parquet contained
  9,700 rows, 30 columns, 42,072 queries, the same universe counts, and no
  license columns.
- Prompt memory carried forward: numbered districts are written without a
  hyphen in Russian (`Дистрикт 12`). Passing model audit is not sufficient for
  this deterministic franchise style rule; search both queries and keywords
  before merge, and reject/repair any violation without manual payload edits.

### Middle-earth batch `me-006` and 10k milestone — kept 2026-08-29

- Original manifest: 300 unique frozen Tolkien Gateway passages, SHA-256
  `321c1a1de791ecbb24f6d25c0f95d7c4b88a452efb7f33ca2463115faf93bc43`.
  Independent high audit accepted 299 originals and rejected one. One unused
  frozen reserve passage about Ulfang passed medium generation and a fresh
  independent high audit; effective manifest SHA-256 is
  `2dbb5e414b34656077da77675cfd7cd387fa6a14a5c5f1ad8d663bcb2a2ba334`.
- Three attempts to audit that single reserve through the Batch API failed in
  pre-execution validation with `total=0`, all token counters zero, and no
  billable usage because the organization could not access newly uploaded
  Batch files. After reproducing the failure with a newly uploaded processed
  file, the Batch path was stopped. One synchronous audit with the identical
  `gpt-5.6-sol`/high request body passed; no model fallback or duplicate
  generation was used.
- Whole-corpus finalization against the 9,700-row baseline found three exact
  normalized-query collisions. Each row passed after one targeted repair and
  an independent high re-audit; final rejects were zero. Final repair
  distribution: 203 rows with zero repairs, 80 with one, 16 with two, and one
  with three. The sole three-repair row, on Elenna's etymology, was manually
  checked against its complete frozen passage and retained.
- The batch ledger contains 837 paid calls: 830 Batch and seven synchronous;
  302 generation, 417 audit, and 118 repair calls. They used 743,020 input and
  770,505 output tokens, including 640,455 reasoning tokens, for an estimated
  `$9.346159`; synchronous calls account for `$0.176532`.
- Canonical corpus after atomic merge: 10,000 rows and 43,872 unique normalized
  queries; 1,808 Middle-earth, 1,600 Harry Potter, 1,300 Hunger Games, and 300
  Witcher rows. All 4,708 new rows have six questions and the exact generator
  model. Exact IDs and content are unique, no license fields exist, and all
  required source provenance fields are populated. Dataset SHA-256:
  `f6aa99df06b50a6101275e4abaff13f0fdd714724fa29ac5fe5a6229ffe84911`.
- The global five-word-shingle pass considered 133,425 nonzero-overlap
  candidate pairs and 58,763 size-eligible pairs. Maximum Jaccard similarity
  was `0.768240343`; zero pairs met or exceeded the `0.78` rejection threshold.
- Full project verification after the loader regression fix: 691 passed, one
  skipped. Scratch pipeline: 58 passed. Polars and `DatasetRecordAdapter`
  materialized all 10,000 rows and preserved the 30-column canonical schema
  and provenance. A regression test now prevents late JSON fields from being
  dropped by Polars' default 100-row inference window.
- Private Hugging Face commit:
  `f295ceffad7938fe473e64384d724674139007e1`; the authenticated remote raw
  download is byte-identical to the local canonical file. Dataset Viewer
  reported `pending=[]`, `failed=[]`, and `partial=false`; authenticated
  server-side Parquet contained 10,000 rows, 30 columns, 43,872 queries, the
  exact universe counts above, and no license columns. Its `default/train`
  final row matched the local `chunk_id`, universe, title, and source URL.
- Prompt memory carried forward: when a compact factual passage contrasts an
  instigator with the operational leader, keep both roles explicit; for older
  legendarium versions, preserve the named text and chronology rather than
  collapsing changes across drafts.

## Image cleanup and retrieval benchmark v1 — published 2026-08-29

### Canonical cleanup

- Removed only `has_image` and `img_path` from all 10,000 canonical records.
  Structural comparison against Git commit `6d5b4b3` after projecting away
  those two keys proved every other field, nested value, row, and ordering
  unchanged. Deleted all 77 Git-tracked files below `.data/img/`; no image
  path or image key remains.
- The cleaned canonical JSON has 10,000 rows, 43,872 globally unique normalized
  queries, 28 top-level columns, and SHA-256
  `323cdffbab56ba6d0fbee62e3d465ad859a191122bf38eebd52951afc020b27d`.
  Source Git commit:
  `3181e599baab142f8c9cd638f35d10c3f0aefc72`.
- The private `justatom/polaroids-ai-retrieval` dataset was replaced at Hub
  commit `c35835414d6a70b5cf579a3b5b8390423aaa9b73`. Its authenticated raw
  download is byte-identical. Viewer `default/train` is 10,000 x 28 and its
  Parquet SHA-256 is
  `21f1bcfbd6c3c28b66cccdf5b2fa3e64f553005b7ce0ac12cf70c031d03d04ab`.

### Relational retrieval representation

- Private Hub dataset: `justatom/universe-retrieval-benchmark`, commit
  `65fb0a7c0ba4b7d53017ce580f3bffd7d7ee3c31`.
- `corpus/corpus`: 10,000 rows with `corpus_id`, `title`, exact source `text`,
  and lossless nested `metadata`.
- `queries/train`: 30,710 rows; `queries/test`: 13,162 rows. Query rows do not
  contain `corpus_id`, `positive_id`, or hard negatives.
- `qrels/train`: 30,710 rows; `qrels/test`: 13,162 rows. Each query has exactly
  one matching qrel with score 1 and a valid shared-corpus ID.
- v1 has no validation split and no fixed negative labels. Retrieval ranks
  against the complete 10,000-passage corpus. QA remains a later derived
  benchmark rather than being mixed into this release.

### Deterministic IDs and leakage-safe split

- Query IDs are UUIDv5 over the normalized query: NFKC, collapsed whitespace,
  trim, and Unicode casefold. All 43,872 normalized queries and IDs are unique.
- Source-group IDs are UUIDv5 over compact JSON descriptors. Wiki rows group
  by normalized `source_name` plus `source_page_id`; legacy rows group by
  normalized `type`, `author`, and `title`.
- Groups are ordered by SHA-256 of seed
  `justatom-universe-retrieval-v1-split-2026-08-29` plus group ID. Stable
  subset-sum reaches the exact 13,162-query test target with 503 whole groups;
  the remaining 30,710 queries use 1,360 groups. The group sets are disjoint.
- Aligned source `query_languages` values are authoritative. Otherwise the
  greater Cyrillic/Latin letter count yields `ru`/`en`, an equal nonzero count
  yields `mixed`, and zero letters yields `unknown`. The final benchmark has
  24,530 English and 19,342 Russian queries.

| Split | Queries | EN | RU | Legacy-source | Wiki-source | Groups |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 30,710 | 17,035 | 13,675 | 8,816 | 21,894 | 1,360 |
| test | 13,162 | 7,495 | 5,667 | 5,008 | 8,154 | 503 |

| Universe | Train queries | Test queries | Total |
| --- | ---: | ---: | ---: |
| legacy/null | 8,816 | 5,008 | 13,824 |
| The Witcher | 1,140 | 660 | 1,800 |
| Harry Potter | 7,050 | 2,550 | 9,600 |
| Middle-earth | 8,100 | 2,748 | 10,848 |
| The Hunger Games | 5,604 | 2,196 | 7,800 |

### Verification and operational lessons

- Two independent builds with fixed audit timestamp produced byte-identical
  README, manifest, and five JSONL files. The independent verifier checked all
  corpus values, UUIDs, queries, qrels, splits, language/universe totals,
  absence of image fields/links, and source-group isolation.
- Local Hugging Face loads passed for all three configs. The corpus card uses
  a 64 MiB JSON `chunksize`, larger than its 37.7 MB JSONL, so schema inference
  sees the complete heterogeneous legacy/wiki metadata union without changing
  any source values.
- All seven raw Hub payload files matched local bytes. All five benchmark
  Viewer Parquet files and the cleaned canonical Parquet were downloaded with
  authentication and compared row-for-row. Arrow's JSON extension represents
  nested JSON objects as encoded strings in `to_pylist()`; decode those before
  semantic equality checks.
- Benchmark Viewer Parquet SHA-256 values: corpus
  `bb9a0bb2cf0e8f6085b9ee9c8cba963c5eeecd7424a60c9d2c234855c4905ac5`;
  queries train
  `4bccbd531c4d8b519e87f5d5f4c962dcfb3887f39843b8d13f8428b4ce55f5e5`;
  queries test
  `184cf2baa4842396145153c8bbfe2422cd48c7bc0faf1354d685e2db05681f17`;
  qrels train
  `34a4ba52d22baeda07013e62b08e6297224d21d158baf37e45d25e05db2e1698`;
  qrels test
  `edeffdf51500108417ce038109a06b4d90d4a51689e991e3ebb5a21dca5cc410`.
- Hub `preview`, `viewer`, `search`, `filter`, and `statistics` all completed
  successfully. This phase made no Responses API or other paid LLM calls;
  estimated additional LLM cost is exactly USD 0.

## Retrieval benchmark v1.1 master table — published 2026-08-30

### Canonical one-row-per-passage view

- The same private `justatom/universe-retrieval-benchmark` repository now has
  Hub commit `7752739f118344590ca561135b07e0eeccbbae18`. The recommended
  `benchmark/full` config contains exactly 10,000 rows and one row per passage.
- Each row retains every image-free non-query Polaroids field at top level and
  replaces the parallel query arrays with `queries: list[struct]`. Each struct
  has the exact source question plus stable `query_id`, `language`, and original
  `query_index`. The row adds only `source_group_id` and `split`.
- The master contains 43,872 nested queries, 9,917 passages with queries, and
  83 empty query lists. Passage allocation is 6,825 train and 3,175 test; empty
  passages inherit their queried source group's split rather than receiving a
  synthetic query or third split.
- The master is the convenient canonical view for Viewer, Polars, filtering,
  slicing, and training-data preparation. The standard `corpus`, `queries`, and
  `qrels` configs remain deterministic compatibility views for BEIR/MTEB-style
  evaluators, avoiding both duplicated passages and ecosystem lock-in.

### Backward compatibility and hashes

- The v1.1 build is master-first: all relational rows are regenerated only
  from master rows. A build gate compared all five derived JSONL files with
  verified v1 bytes; zero files changed. Their server-side Viewer Parquet
  hashes also stayed byte-identical to v1.
- Master JSONL SHA-256:
  `b22bd1b332250612898d873aec074c2e3e111a2419ad5551aeb8f0664e905b13`
  (`47,564,699` bytes). Viewer `benchmark/full` Parquet SHA-256:
  `95da61bcb4560e7c3ec4f2c39052116a5b7fb3a9a3ae5a7f3fb5ea2ab2582948`
  (`42,967,658` bytes).
- The cleaned source stayed immutable at SHA-256
  `323cdffbab56ba6d0fbee62e3d465ad859a191122bf38eebd52951afc020b27d`
  and last-touch Git commit
  `3181e599baab142f8c9cd638f35d10c3f0aefc72`. No image key, image path,
  answer, hard negative, model score, or license field was introduced.

### Verification and operating memory

- Two full builds with fixed audit timestamp produced the same eight files.
  An independent verifier reconstructed IDs, languages, groups, splits, master
  rows, corpus rows, queries, and qrels directly from the source without
  importing the production builder. It reported 29 master columns, 1,360/503
  disjoint train/test groups, and zero compatibility changes.
- Scratch tests reached 17 passing tests. Polars and Hugging Face `datasets`
  loaded all six physical JSONL splits with exact counts and lossless first/last
  values. Tests use a fresh local HF cache because `datasets` otherwise reuses
  the same `artifact/<config>` fingerprint across temporary fixture contents.
- The authenticated raw Hub download matched all eight local payload files.
  Dataset Viewer reached `pending=[]`, `failed=[]`, `partial=false`; all six
  logical partitions compared row-for-row, including every nested struct and
  all 83 empty arrays. Authenticated first and last rows matched for all six
  partitions, and preview/viewer/search/filter/statistics were green.
- The remote verifier groups and lexically orders physical Parquet shards by
  `(config, split)` before comparison, so future multi-shard Viewer output will
  not be mistaken for multiple logical splits. Dataset Viewer filter predicates
  require quoted column identifiers, for example `"score"=1`.
- This v1.1 phase made zero paid LLM calls and cost USD 0. QA generation and
  independently audited hard-negative mining remain separate future phases;
  neither mutates this 10,000-row retrieval checkpoint.

## Repository distribution — 2026-09-05

- Git no longer carries a corpus payload. The `justatom` preset streams the
  private Hub `benchmark/full` view at commit
  `7752739f118344590ca561135b07e0eeccbbae18`.
- The Hub master view contains 10,000 passages and 43,872 structured queries,
  with stable `source_group_id` values and leakage-safe row-level `split`
  assignments. BEIR-like `corpus`, `queries`, and `qrels` views remain in the
  same private dataset.
- The previously prepared 4,992-row legacy JSONL slice was byte-validated
  (SHA-256
  `0e54a7dfb7ff973a777e1303f6b95641551e0b77875c5813c751137a1439f176`)
  but is intentionally not distributed through Git because the Hub is the
  canonical source.
- Image and license fields and image-path strings are absent from the Hub
  benchmark. All 77 formerly tracked dataset images were removed. Local
  `llms.txt` development memory remains unversioned.
