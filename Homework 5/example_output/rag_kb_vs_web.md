# RAG: визначення й компоненти (за базою знань) vs. підходи 2026 року (за свіжими веб-джерелами)

Досліджено: (1) що таке retrieval-augmented generation (RAG) і з яких компонентів воно складається — **за документами з локальної бази знань**; (2) які **підходи до RAG обговорюються у 2026 році** — за **свіжими інтернет-джерелами**, відкритими й прочитаними під час пошуку.

## 1) Що таке RAG і які його компоненти (за документами у базі знань)

### Визначення
RAG (retrieval-augmented generation) у документах бази знань описується як техніка/підхід, що **поєднує LLM із системою пошуку/витягання документів**, аби модель могла **отримувати та включати нову інформацію з зовнішніх джерел** під час відповіді. Тобто перед генерацією відповідь “підкріплюється” знайденими документами, які доповнюють знання з тренувальних даних моделі. [retrieval-augmented-generation.pdf p.1], [large-language-model.pdf p.8]

### Базовий процес (pipeline)
У базі знань RAG описується як процес, де:
1) для документів (корпусу) формуються векторні подання (embeddings) та зберігаються у **векторній базі даних**;
2) на запит користувача викликається **retriever**, який відбирає релевантні документи;
3) релевантна інформація додається до промпту й передається в LLM для генерації відповіді. [retrieval-augmented-generation.pdf p.2]

### Компоненти RAG (мінімальний набір, який прямо згадується в KB)
З урахуванням формулювань у документах KB, можна виділити такі ключові компоненти:

- **Зовнішнє джерело знань / набір документів (corpus)**: “specified set of documents”, який використовується як додаткове джерело інформації. [retrieval-augmented-generation.pdf p.1]
- **Embedding-представлення (векторизація)**: документи подаються як вектори у великому векторному просторі (embeddings). [retrieval-augmented-generation.pdf p.2]
- **Векторне сховище / vector database**: embeddings зберігаються у векторній БД для подальшого пошуку. [retrieval-augmented-generation.pdf p.2]
- **Document retriever (модуль витягання/пошуку)**: за запитом знаходить “most relevant documents”. [retrieval-augmented-generation.pdf p.2], [large-language-model.pdf p.8]
- **Generator / LLM (модуль генерації)**: отримує запит, доповнений витягнутими документами, і генерує відповідь. (У KB це описано як “integrates LLMs with document retrieval systems” та як збагачення промпту зовнішніми документами.) [large-language-model.pdf p.8], [retrieval-augmented-generation.pdf p.2]

### Додаткові/поширені компоненти, які також згадані в KB
- **Reranking (переранжування)**: у KB згадується, що reranking-техніки можуть підсилювати retriever, пріоритизуючи найбільш релевантні документи серед витягнутих. [retrieval-augmented-generation.pdf p.3]

## 2) Які підходи до RAG обговорюються у 2026 році (за веб-джерелами)

Нижче — зріз тем і “архітектурних напрямів”, які у 2026-му описуються як розвиток RAG за прочитаними веб-джерелами.

### 2.1 Agentic RAG (агентний RAG, багатоетапне керування retrieval)
Оглядовий survey 2026 року позиціонує **Agentic RAG** як еволюцію від статичних/лінійних RAG-пайплайнів до систем, де автономні агенти (на базі LLM) **планують**, **використовують інструменти**, **відбивають/перевіряють проміжні кроки**, **ітеративно уточнюють контекст** і **динамічно керують стратегіями retrieval**. Це подається як відповідь на обмеження “traditional RAG” зі статичним workflow та слабкою придатністю до multi-step reasoning. https://arxiv.org/html/2501.09136v4

Конкретні інженерні патерни агентності (tool use, planning тощо) підкреслюються як ключ до адаптивності. https://arxiv.org/html/2501.09136v4

### 2.2 “Harness поверх enterprise search”: агентність без перебудови індексації
Практична робота (2026) описує AgenticRAG як “легку обв’язку” (harness) над уже наявною enterprise search-інфраструктурою. Ідея: зменшити залежність від одноразово сформованого candidate set, надавши LLM інструменти **search / find / open / summarize** для ітеративного пошуку, навігації в документах і консолідації доказів (з урахуванням токен-бюджету). https://arxiv.org/html/2605.05538v1

Це репрезентує тренд 2026 року: RAG як **контур керування пошуком + читанням документів**, а не тільки “vector top-k → промпт → відповідь”. https://arxiv.org/html/2605.05538v1

### 2.3 Наївний/модульний/графовий RAG як «еволюційна лінійка» до агентності
У survey про Agentic RAG зазначено, що еволюція RAG-парадигм часто описується як перехід від **naïve** до **modular** та **graph-based RAG**, а далі — до agentic систем (термінологія і класифікація можуть різнитися між авторами). https://arxiv.org/html/2501.09136v4

### 2.4 Пам’ять і long-document RAG (довгі документи, багатокрокове накопичення контексту)
Оглядовий матеріал 2026 року акцентує, що “advanced RAG” рухається від простого векторного пошуку до **long-document memory** та технік, що допомагають тримати “global view” і зшивати розкидані докази:
- **MiA-RAG**: побудова високорівневого резюме (глобального огляду) довгого документа, яке далі спрямовує retrieval і відповідь.
- **HGMem**: організація витягнутої інформації як гіперграфа для багатокрокової пам’яті й multi-hop reasoning.
- приклади “дискурс-орієнтованого” підходу (Disco-RAG) для кращого синтезу доказів з урахуванням структури/зв’язків. https://www.turingpost.com/p/ragtypes

### 2.5 Адаптивний retrieval, фільтрація та верифікація (uncertainty/evidence sufficiency)
В 2026-добірці підходів згадуються категорії, де retrieval стає умовним/адаптивним та доповнюється перевірками:
- **QuCo-RAG**: тригерить retrieval для “довгохвостих”/рідкісних сутностей (довідка: опис у добірці як conditional retrieval за статистиками/частотами).
- **SURE-RAG**: фокус на достатності доказів (support/refute/insufficient) і здатності утриматися від відповіді.
- **HiFi-RAG**: багатоступеневе “очищення”/прюнінг контексту перед фінальною генерацією (в добірці як приклад багатостадійного фільтрування). https://www.turingpost.com/p/ragtypes

Також практична робота з AgenticRAG прямо обговорює перехід від single-shot retrieval до ітеративного tool use, а також multi-query пошук і in-document navigation як фактори якості/ефективності. https://arxiv.org/html/2605.05538v1

### 2.6 Bidirectional / write-back RAG (контрольоване поповнення знань)
У 2026-добірці фігурує **Bidirectional RAG** як підхід із контрольованим “write-back” у корпус: згенеровані відповіді додаються лише після grounding-перевірок (наприклад entailment/attribution/novelty). https://www.turingpost.com/p/ragtypes

## Comparison

| Вісь порівняння | KB-документи (визначення/компоненти) | Веб-джерела 2026 (підходи/тренди) |
|---|---|---|
| Базова ідея | LLM + document retrieval; документи доповнюють знання моделі [retrieval-augmented-generation.pdf p.1] | RAG як “шар” reasoning/memory/governance навколо LLM; вихід за межі простого retrieve-then-generate https://www.turingpost.com/p/ragtypes |
| Мінімальні компоненти | corpus → embeddings → vector DB → retriever → LLM generator [retrieval-augmented-generation.pdf p.2] | додаються керування стратегіями retrieval, інструменти, пам’ять, верифікація; інколи графи/структури https://arxiv.org/html/2501.09136v4 |
| Роль reranking / фільтрації | згадується reranking для покращення retriever [retrieval-augmented-generation.pdf p.3] | багатоступеневе фільтрування, conditional retrieval, evidence sufficiency/abstention; ітеративність https://www.turingpost.com/p/ragtypes |
| Динамічність workflow | здебільшого лінійний pipeline “retrieve → augment prompt → generate” [retrieval-augmented-generation.pdf p.2] | agentic/iterative retrieval, multi-query, navigation всередині документів, summarize для керування контекстом https://arxiv.org/html/2605.05538v1 |

## Conclusions

- За KB, RAG — це інтеграція **LLM** із **retriever**, який дістає релевантні документи з **зовнішнього корпусу**, часто через **embeddings + vector database**, і підкладає їх у контекст для генерації відповіді; як підсилення може застосовуватись **reranking**. [retrieval-augmented-generation.pdf p.1], [retrieval-augmented-generation.pdf p.2], [retrieval-augmented-generation.pdf p.3]
- За веб-джерелами 2026 року, акцент зміщується до: **Agentic RAG** (планування, tool use, ітеративний retrieval, багатoагентність), **long-document/memory RAG** (глобальний огляд, структурована пам’ять), **verification/uncertainty-aware RAG**, а також більш практичних “harness” підходів для enterprise пошуку без повної перебудови індексації. https://arxiv.org/html/2501.09136v4 ; https://arxiv.org/html/2605.05538v1 ; https://www.turingpost.com/p/ragtypes

## Sources
- Knowledge base: retrieval-augmented-generation.pdf p.1, p.2, p.3; large-language-model.pdf p.8; langchain.pdf p.1, p.4
- Web: https://arxiv.org/html/2501.09136v4 ; https://arxiv.org/html/2605.05538v1 ; https://www.turingpost.com/p/ragtypes ; https://www.edenai.co/post/the-2026-guide-to-retrieval-augmented-generation-rag
