# RAG (Retrieval-Augmented Generation): визначення, компоненти (за локальною базою) та підходи, актуальні у 2026 (за веб-джерелами)

Досліджено (1) що таке Retrieval-Augmented Generation (RAG) і з яких компонентів складається за документами з локальної бази знань; (2) які підходи/варіації RAG обговорюються у 2026 році за свіжими відкритими джерелами в інтернеті (переважно огляди/таксономії та статті 2025–2026).

## 1) Що таке RAG — за документами у базі знань

У локальних документах RAG визначається як техніка, що **дозволяє великим мовним моделям (LLM) “діставати” (retrieve) нову інформацію із зовнішніх джерел даних і включати її у відповідь**, тобто перед генерацією модель спирається на підмножину заданих документів, які доповнюють знання з пре-тренування моделі. Це підвищує актуальність/доменно-специфічність відповідей. [retrieval-augmented-generation.pdf p.1]

Також RAG описано як підхід, що **інтегрує LLM із системами пошуку документів**: за запитом викликається документний ретрівер, який повертає релевантні документи, після чого вони використовуються у генерації. [large-language-model.pdf p.8]

## 2) Компоненти RAG (локальна база знань)

Згідно з локальними матеріалами, базова архітектура/конвеєр RAG включає такі сутності (у термінах документів):

- **Зовнішнє джерело знань/корпус документів** (specified set of documents), до якого система звертається перед відповіддю. [retrieval-augmented-generation.pdf p.1]
- **Представлення документів у вигляді векторних ембедингів** (embeddings) у “великому векторному просторі”; RAG може працювати з неструктурованими, напівструктурованими або структурованими даними (включно з knowledge graphs). [retrieval-augmented-generation.pdf p.2]
- **Векторна база даних (vector database)** для зберігання ембедингів та підтримки пошуку/ретріва. [retrieval-augmented-generation.pdf p.2]
- **Document retriever (ретрівер)**: за запитом користувача обирає найбільш релевантні документи (порівняння може робитися різними методами залежно від індексації). [retrieval-augmented-generation.pdf p.2]
- **LLM / генератор (generator)**: отримує запит, “аугментований” (доповнений) знайденими фрагментами/документами, і генерує відповідь. Опис процесу подано як комбінування зовнішніх документів і user input у prompt для отримання відповіді. [retrieval-augmented-generation.pdf p.2]

Додатково локальні документи згадують типові елементи/техніки, які часто є частиною практичних RAG-систем:

- **Chunking**: стратегії розбиття даних на фрагменти (chunks) для векторизації і подальшого пошуку деталей. [retrieval-augmented-generation.pdf p.4]
- **Reranking** (переранжування): техніки для покращення роботи ретріверу шляхом пріоритизації найбільш релевантних отриманих документів. [retrieval-augmented-generation.pdf p.3]

> Примітка: у видачі локальної бази частина пасажів про “pipeline components indexing/chunking/reranker” була позначена як **нижче порогу релевантності**, але твердження, наведені вище, підтверджені прямими цитованими фрагментами з того ж документа. [retrieval-augmented-generation.pdf p.2] [retrieval-augmented-generation.pdf p.3] [retrieval-augmented-generation.pdf p.4]

## 3) Які підходи до RAG обговорюються у 2026 році — за свіжими веб-джерелами

Нижче — підходи/архітектурні напрямки, які фігурують у публікаціях 2025–2026 та в оглядових матеріалах, доступних у вебі.

### 3.1 Agentic RAG (агентний/agentic підхід)

Огляд 2026 року (arXiv HTML-версія) позиціонує **Agentic RAG** як еволюцію над “традиційним” RAG із **статичним, лінійним workflow**, де **автономні агенти** додають контрольний шар: планування, tool-use, reflection, (multi-)agent collaboration; це дає можливість **динамічно керувати стратегіями ретріва, ітеративно уточнювати контекст і адаптувати workflow**. У тексті також прямо згадується еволюція парадигм RAG, зокрема “naïve, modular, and graph-based RAG” у контексті переходу до agentic систем. https://arxiv.org/html/2501.09136v4

Суміжне пояснення “agentic RAG vs traditional RAG” як **ітеративного циклу plan→route→iterate→check** наведено у практичному гіді (2026) з переліком патернів (router agent, agent-as-retriever, corrective RAG, self-RAG, adaptive RAG, agentic GraphRAG). https://www.lyzr.ai/blog/agentic-rag/

### 3.2 Self-RAG / Corrective RAG (саморефлексивні та «коригувальні» варіанти)

Оглядовий матеріал про “what’s new in RAG” описує **Self-RAG** як підхід, у якому модель **динамічно вирішує, коли робити retrieval і чи використовувати retrieved content**, керуючись спеціальними self-reflection tokens (для тригеру ретріва, оцінки релевантності, критики власного output). Це подається як спосіб зменшення hallucinations і підвищення фактичної точності. https://www.codecon.sk/whats-new-in-rag/

### 3.3 GraphRAG / Knowledge-graph RAG (графові індекси та багатохопне узагальнення)

У тому ж огляді описано **GraphRAG** як графо-орієнтований підхід для задач, що потребують комбінування інформації з багатьох документів: LLM будує двоетапний graph index (entity graph + “community summaries”), а відповідь збирається через часткові відповіді від спільнот і фінальне узагальнення. https://www.codecon.sk/whats-new-in-rag/

Окремо, практичний “agentic RAG” гід виокремлює **Agentic GraphRAG** як патерн, де агент виконує traversal знаннєвого графа для зв’язків між сутностями (multi-hop). https://www.lyzr.ai/blog/agentic-rag/

### 3.4 ChunkRAG, HtmlRAG, Multimodal RAG (опрацювання форматів/структури та мультимодальність)

У переліку “what’s new” згадуються:

- **ChunkRAG**: акцент на розумнішому chunking/filtering (щоб зменшити нерелевантний контекст, пришвидшити доступ, знизити hallucinations). https://www.codecon.sk/whats-new-in-rag/
- **HtmlRAG**: ідея використовувати HTML-структуру як знання (з очищенням/прунингом блоків DOM), щоб не втрачати семантику структури веб-сторінок при перетворенні в plain text. https://www.codecon.sk/whats-new-in-rag/
- **Multimodal RAG**: розширення retrieval на кілька модальностей (текст+зображення+аудіо+відео). https://www.codecon.sk/whats-new-in-rag/

### 3.5 «Типології» RAG у практичних гайдах (спрощені таксономії)

Комерційно-орієнтований гайд (2025) узагальнює низку “типів RAG” (simple RAG, RAG with memory, agentic RAG, graph RAG, self-RAG, branched RAG, multimodal RAG, adaptive RAG тощо) і пояснює їх на рівні архітектурних ідей та компромісів (latency/cost vs якість). Це корисно як «ринкова» таксономія, але не є академічним джерелом. https://www.meilisearch.com/blog/rag-types

## Comparison: базовий RAG vs підходи, що популяризуються у 2026

| Підхід | Ключова ідея | Яку проблему адресує | Джерело |
|---|---|---|---|
| “Класичний” pipeline RAG | Одноразовий retrieval релевантних документів + генерація | Актуальність/фактичність через зовнішні документи | [retrieval-augmented-generation.pdf p.1], [large-language-model.pdf p.8] |
| Agentic RAG | Агент(и) керують retrieval як циклом (planning/routing/verification) | Multi-step задачі, багатоджерельність, адаптивність workflow | https://arxiv.org/html/2501.09136v4 ; https://www.lyzr.ai/blog/agentic-rag/ |
| Self-RAG / Corrective RAG | Самоперевірка та рішення “коли/чи треба retrieve”, критика відповіді | Hallucinations, нерелевантний/слабкий retrieval | https://www.codecon.sk/whats-new-in-rag/ ; https://www.lyzr.ai/blog/agentic-rag/ |
| GraphRAG | Індекс як граф сутностей + узагальнення по “спільнотах” | Багатодокументні питання, multi-hop зв’язки | https://www.codecon.sk/whats-new-in-rag/ |
| HtmlRAG | Використання HTML-структури (DOM) як носія знання | Втрата структури при HTML→text та шум raw HTML | https://www.codecon.sk/whats-new-in-rag/ |
| Multimodal RAG | Retrieval по різних модальностях | Запити, де потрібні зображення/аудіо/відео | https://www.codecon.sk/whats-new-in-rag/ |

## Conclusions

- За локальною базою знань, RAG — це **інтеграція LLM з ретрівом**: документи → ембединги → зберігання у векторній БД → ретрівер дістає релевантні фрагменти → LLM генерує відповідь на основі запиту + контексту. Основні компоненти: **джерело документів, embeddings, vector database, retriever, LLM/generator**, а також типові практики на кшталт **chunking** і **reranking**. [retrieval-augmented-generation.pdf p.1] [retrieval-augmented-generation.pdf p.2] [retrieval-augmented-generation.pdf p.3] [retrieval-augmented-generation.pdf p.4] [large-language-model.pdf p.8]
- За веб-джерелами 2026 року, центральний тренд — **Agentic RAG** (і споріднені патерни), тобто перехід від статичного конвеєра до **динамічного керування retrieval** (планування, маршрутизація між джерелами, ітерації, верифікація/самокритика). https://arxiv.org/html/2501.09136v4
- Паралельно активно обговорюються спеціалізовані варіанти: **Self-RAG/Corrective RAG** (саморефлексія і контроль якості контексту), **GraphRAG** (графові індекси для multi-hop), а також інженерні напрямки на кшталт **HtmlRAG, ChunkRAG, Multimodal RAG**. https://www.codecon.sk/whats-new-in-rag/

## Sources
- Knowledge base: retrieval-augmented-generation.pdf p.1, p.2, p.3, p.4; large-language-model.pdf p.8; langchain.pdf p.4
- Web: https://arxiv.org/html/2501.09136v4 ; https://www.codecon.sk/whats-new-in-rag/ ; https://www.meilisearch.com/blog/rag-types ; https://www.lyzr.ai/blog/agentic-rag/