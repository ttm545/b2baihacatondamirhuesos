import { createIndex, evaluate, readCorpus } from "./engine.mjs";
import { demo } from "./demo.mjs";

const ui = Object.fromEntries([...document.querySelectorAll("[id]")].map(node => [node.id, node]));
const element = (tag, text = "", className = "") => {
  const node = document.createElement(tag);
  node.textContent = text;
  if (className) node.className = className;
  return node;
};
const formatNumber = new Intl.NumberFormat("ru");
const plurals = new Intl.PluralRules("ru");
const decimal = new Intl.NumberFormat("ru", { maximumFractionDigits: 2 });
const percent = new Intl.NumberFormat("ru", { style: "percent", maximumFractionDigits: 1 });
const countLabel = (count, forms) => `${formatNumber.format(count)} ${forms[plurals.select(count)] ?? forms.many}`;
let corpus, index, report;
let busy = false;

function setBusy(value) {
  busy = value;
  ui["import-button"].disabled = value;
  ui["run-bench"].disabled = value;
  ui.k.disabled = value;
}
function notice(text = "", error = false) {
  ui.notice.textContent = text;
  ui.notice.hidden = !text;
  ui.notice.classList.toggle("error", error);
}
function openPanel(search) {
  ui["search-panel"].hidden = !search;
  ui["bench-panel"].hidden = search;
  ui["search-tab"].classList.toggle("active", search);
  ui["bench-tab"].classList.toggle("active", !search);
  ui["search-tab"].setAttribute("aria-pressed", search);
  ui["bench-tab"].setAttribute("aria-pressed", !search);
  ui["page-title"].textContent = search ? "Проверка поиска" : "Бенчмарк поиска";
  ui["page-description"].textContent = search
    ? "Проверьте выдачу и откройте найденное письмо."
    : "Запустите оценку на размеченных запросах и выгрузите выдачу.";
}
function load(data, name) {
  const prepared = createIndex(data.documents);
  corpus = data;
  index = prepared;
  ui["data-name"].textContent = name;
  const documentCount = countLabel(data.documents.length, { one: "письмо", few: "письма", many: "писем" });
  const queryCount = countLabel(data.queries.length, { one: "запрос", few: "запроса", many: "запросов" });
  ui["data-counts"].textContent = `${documentCount} · ${queryCount}`;
  ui.mailbox.replaceChildren(...index.mailboxes.map(name => new Option(name, name)));
  ui.example.replaceChildren(new Option("Выбрать запрос", ""), ...data.queries.map((item, i) => new Option(item.query, i)));
  ui.query.value = "";
  ui["as-of"].value = "";
  resetBenchmark();
  renderSearch();
}
function resetBenchmark() {
  report = null;
  ui.export.disabled = true;
  ui["bench-progress"].textContent = "";
  ui.metrics.replaceChildren();
  ui["bench-rows"].replaceChildren(element("p", "Запустите проверку, чтобы увидеть метрики.", "empty"));
}
function letterParts(text) {
  const normalized = text.trim();
  const firstLine = normalized.split(/\r?\n/, 1)[0] || "(Пустое письмо)";
  const explicit = /^(?:тема|subject)\s*:\s*(.+)$/im.exec(normalized);
  const title = (explicit?.[1] || firstLine).trim().slice(0, 110);
  const body = explicit || firstLine.length > 110 ? normalized : normalized.slice(firstLine.length).trimStart();
  return { title, body };
}
function showLetter(hit, selected, buttons) {
  buttons.forEach(btn => btn.setAttribute("aria-pressed", btn === selected));
  const { title, body } = letterParts(hit.doc.doc);
  const article = element("article", "", "letter-content");
  article.append(element("h3", title),
    element("p", `${hit.doc.doc_dt || "Дата неизвестна"} · ${hit.doc.synthetic_user} · ID ${hit.doc.doc_id}`, "meta"),
    element("p", body || "В этом письме нет основного текста.", "letter-body"));
  ui.letter.replaceChildren(article);
}
function renderSearch() {
  const start = performance.now();
  const query = ui.query.value.trim();
  const hits = index.search(query, ui.mailbox.value, ui["as-of"].value);
  ui["results-title"].textContent = query ? "Результаты поиска" : "Письма";
  ui["search-summary"].textContent = `${hits.length}${hits.length === 25 ? "+" : ""} · ${decimal.format(performance.now() - start)} мс`;
  ui.results.replaceChildren();
  if (!hits.length) {
    ui.results.append(element("p", "Ничего не найдено. Попробуйте изменить запрос или фильтры.", "empty"));
    ui.letter.replaceChildren(element("p", "Здесь откроется найденное письмо.", "empty"));
    return;
  }
  const buttons = hits.map(hit => {
    const { title, body } = letterParts(hit.doc.doc);
    const button = element("button", "", "result-button");
    button.type = "button";
    const meta = element("span", "", "meta");
    meta.append(element("span", hit.doc.doc_dt || "Дата неизвестна"));
    if (query) meta.append(element("span", decimal.format(hit.score)));
    button.append(meta, element("strong", title), element("span", body.replace(/\s+/g, " "), "snippet"));
    button.addEventListener("click", () => showLetter(hit, button, buttons));
    return button;
  });
  ui.results.append(...buttons);
  showLetter(hits[0], buttons[0], buttons);
}
function useQuery(item) {
  ui.query.value = item.query;
  ui.mailbox.value = String(item.synthetic_user);
  ui["as-of"].value = item.query_date;
  openPanel(true);
  renderSearch();
}
function showReport(result, k) {
  const metricItems = [
    [`Recall@${k}`, result.recall], [`nDCG@${k}`, result.ndcg], [`MRR@${k}`, result.mrr],
    ["p99 в браузере", `${decimal.format(result.p99)} мс`]
  ];
  const metricValue = value => typeof value === "number" ? percent.format(value) : value ?? "—";
  ui.metrics.replaceChildren(...metricItems.map(([name, value]) => {
    const card = element("div", "", "metric");
    card.append(element("span", name), element("strong", metricValue(value)));
    return card;
  }));
  const table = element("table");
  const head = table.createTHead().insertRow();
  for (const name of ["Запрос", "Ящик", `Recall@${k}`, `nDCG@${k}`, `MRR@${k}`]) head.append(element("th", name));
  const body = table.createTBody();
  result.rows.forEach(({ query, recall, ndcg, mrr }) => {
    const row = body.insertRow();
    const link = element("button", query.query, "query-link");
    link.type = "button";
    link.addEventListener("click", () => {
      ui.example.value = String(corpus.queries.indexOf(query));
      useQuery(query);
    });
    row.insertCell().append(link);
    for (const value of [query.synthetic_user, metricValue(recall), metricValue(ndcg), metricValue(mrr)])
      row.insertCell().textContent = value;
  });
  ui["bench-rows"].replaceChildren(table);
  ui["bench-progress"].textContent = `Оценено ${result.scored} из ${result.rows.length} запросов.`;
  ui.export.disabled = false;
}
ui["search-tab"].addEventListener("click", () => openPanel(true));
ui["bench-tab"].addEventListener("click", () => openPanel(false));
ui["search-form"].addEventListener("submit", event => { event.preventDefault(); renderSearch(); });
[ui.mailbox, ui["as-of"]].forEach(control => control.addEventListener("change", () => {
  ui.example.value = "";
  renderSearch();
}));
ui.example.addEventListener("change", () => {
  if (ui.example.value !== "") useQuery(corpus.queries[Number(ui.example.value)]);
});
ui["import-button"].addEventListener("click", () => ui["import-file"].click());
ui["import-file"].addEventListener("change", async () => {
  const [file] = ui["import-file"].files;
  if (!file || busy) return;
  setBusy(true);
  notice("Читаем архив и строим индекс…");
  try {
    const data = await readCorpus(file);
    load(data, file.name);
    notice("Архив загружен. Можно искать письма или запускать бенчмарк.");
  } catch (error) {
    notice(`Архив не загружен. ${error.message}`, true);
  } finally {
    ui["import-file"].value = "";
    setBusy(false);
  }
});
ui.k.addEventListener("change", resetBenchmark);
ui["run-bench"].addEventListener("click", async () => {
  if (busy) return;
  setBusy(true);
  resetBenchmark();
  const k = Number(ui.k.value);
  ui["bench-progress"].textContent = "Считаем метрики…";
  try {
    report = await evaluate(index, corpus.queries, corpus.judgments, k, done => {
      ui["bench-progress"].textContent = `Проверено ${done} из ${corpus.queries.length} запросов…`;
    });
    showReport(report, k);
  } catch (error) {
    ui["bench-progress"].textContent = `Ошибка при проверке. ${error.message}`;
  } finally {
    setBusy(false);
  }
});
ui.export.addEventListener("click", () => {
  if (!report) return;
  const lines = report.rows.map(({ query, ids }) => JSON.stringify({ query_id: query.query_id, doc_ids: ids }));
  const url = URL.createObjectURL(new Blob([lines.join("\n") + "\n"], { type: "application/x-ndjson" }));
  const link = element("a");
  link.href = url;
  link.download = "predictions.jsonl";
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
});
load(demo, "Демо-набор");
ui.example.value = "1";
useQuery(demo.queries[1]);
