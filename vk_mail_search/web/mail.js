const el = id => document.getElementById(id);
let view = 0;
let reading = 0;
let offset = 0;
let topic = "";
let lastQuery = "";
const shortcuts = Array.from(el("shortcuts").querySelectorAll("button"));
const pageSize = 30;

async function get(path) {
  const response = await fetch("api/" + path);
  if (!response.ok) throw new Error("Ошибка запроса (" + response.status + ")");
  return response.json();
}

function highlight(node, text) {
  const prefix = word => word.toLowerCase().replaceAll("ё", "е").slice(0, 5);
  const words = new Set((topic.match(/\p{L}{3,}/gu) || []).map(prefix));
  node.replaceChildren();
  for (const token of text.split(/(\p{L}+)/u)) {
    if (words.has(prefix(token))) {
      const mark = document.createElement("mark");
      mark.textContent = token;
      node.append(mark);
    } else {
      node.append(token);
    }
  }
}

function clearReader(title = "Выберите письмо") {
  ++reading;
  el("similar").hidden = true;
  el("related").replaceChildren();
  el("subject").textContent = title;
  el("stamp").textContent = "";
  el("body").textContent = "";
}

async function openLetter(id, button) {
  clearReader("Загрузка письма…");
  const ticket = reading;
  document.querySelector(".mail[aria-current]")?.removeAttribute("aria-current");
  button.setAttribute("aria-current", "true");
  try {
    const doc = await get("docs/" + encodeURIComponent(id));
    if (ticket !== reading) return;
    el("subject").textContent = doc.subject;
    el("stamp").textContent = doc.synthetic_user + " · " + (doc.doc_dt || "Дата неизвестна");
    highlight(el("body"), doc.doc);
    if (matchMedia("(max-width: 780px)").matches) el("subject").scrollIntoView();
    await loadSimilar(id, ticket);
  } catch (error) {
    if (ticket === reading) clearReader(error.message);
  }
}

async function loadSimilar(id, ticket) {
  try {
    const rows = await get("docs/" + encodeURIComponent(id) + "/similar?k=5");
    if (ticket !== reading) return;
    el("similar").hidden = false;
    for (const row of rows) {
      const link = document.createElement("button");
      link.type = "button";
      link.className = "mail";
      link.textContent = row.subject + " · " + (row.doc_dt || "Без даты");
      link.onclick = () => openLetter(row.doc_id, link);
      el("related").append(link);
    }
    if (!rows.length) el("related").textContent = "Похожих писем в этом ящике нет.";
  } catch (error) {
    if (ticket !== reading) return;
    el("similar").hidden = false;
    el("related").textContent = error.message;
  }
}

function addRow(doc) {
  const button = document.createElement("button");
  button.className = "mail";
  for (const [tag, value] of [["time", doc.doc_dt || "Без даты"], ["strong", doc.subject], ["span", doc.snippet]]) {
    const field = document.createElement(tag);
    if (tag === "span") field.className = "preview";
    if (tag === "span") highlight(field, value);
    else field.textContent = value;
    button.append(field);
  }
  button.onclick = () => openLetter(doc.doc_id, button);
  el("rows").append(button);
}

async function refresh(append = false) {
  const ticket = ++view;
  el("period").hidden = true;
  const mailbox = el("box").value;
  if (!mailbox) return;
  if (!append) {
    offset = 0;
    el("rows").replaceChildren();
    clearReader();
  }
  const query = append ? "" : el("query").value.trim();
  el("more").hidden = true;
  el("status").textContent = "Загрузка…";
  el("results-title").textContent = query ? "Результаты поиска" : "Письма";
  try {
    let rows, status, period, parsedTopic = "";
    if (query) {
      const params = new URLSearchParams({ q: query, mailbox, k: "30" });
      if (el("asof").value) params.set("as_of", el("asof").value);
      const data = await get("search?" + params);
      rows = data.results;
      period = data.period;
      parsedTopic = data.topic;
      status = rows.length ? "Показано: " + rows.length + " · " + data.took_ms + " мс" : "Писем по запросу нет";
    } else {
      rows = await get("mailboxes/" + encodeURIComponent(mailbox) + "/docs?limit=" + pageSize + "&offset=" + offset);
      status = "Показано: " + (offset + rows.length);
    }
    if (ticket !== view) return;
    topic = parsedTopic;
    lastQuery = query;
    if (period || topic) {
      el("period").textContent = [period && "Период: " + period, topic && "Тема: " + topic].filter(Boolean).join(" · ");
      el("period").hidden = false;
    }
    if (period) {
      const inside = rows.filter(doc => doc.in_period);
      const outside = rows.filter(doc => !doc.in_period);
      status = "За период писем: " + inside.length + " (в выдаче) · " + status;
      inside.forEach(addRow);
      if (outside.length) {
        const heading = document.createElement("h3");
        heading.textContent = "Ближе всего по теме, вне периода";
        el("rows").append(heading);
        outside.forEach(addRow);
      }
    } else {
      rows.forEach(addRow);
    }
    offset += rows.length;
    el("more").hidden = Boolean(query) || offset >= Number(el("box").selectedOptions[0].dataset.count);
    el("status").textContent = status;
  } catch (error) {
    if (ticket === view) {
      el("status").textContent = error.message;
      el("more").hidden = !append;
    }
  }
}

for (const shortcut of shortcuts) {
  shortcut.onclick = () => {
    const input = el("query");
    let base = input.value.trim();
    if (base === lastQuery) base = topic;
    else {
      const prefix = shortcuts.map(b => b.dataset.phrase).find(p => base === p || base.startsWith(p + " "));
      if (prefix) base = base.slice(prefix.length).trim();
    }
    input.value = [shortcut.dataset.phrase, base].filter(Boolean).join(" ");
    refresh();
  };
}

el("filters").onsubmit = event => { event.preventDefault(); refresh(); };
function chooseMailbox() {
  const latest = el("box").selectedOptions[0].dataset.latest;
  el("asof").value = latest;
  el("date-hint").textContent = latest ? "Сегодня в этом ящике: " + latest.split("-").reverse().join(".")
                                       : "В ящике нет писем с известной датой";
  return refresh();
}
el("box").onchange = chooseMailbox;
el("more").onclick = () => refresh(true);
async function start() {
  try {
    const boxes = await get("mailboxes");
    for (const { mailbox, count, latest } of boxes) {
      const option = new Option(mailbox + " (" + count + ")", mailbox);
      option.dataset.latest = latest || "";
      option.dataset.count = count;
      el("box").add(option);
    }
    if (boxes.length) await chooseMailbox();
    else el("status").textContent = "В архиве пока нет писем";
  } catch (error) { el("status").textContent = error.message; }
}
start();
