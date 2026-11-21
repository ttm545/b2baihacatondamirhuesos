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

function showPeriodResults(inside, outside, status) {
  el("rows").replaceChildren();
  inside.forEach(addRow);
  if (outside.length) {
    const heading = document.createElement("h3");
    heading.textContent = "Ближе всего по теме, вне периода";
    el("rows").append(heading);
    outside.forEach(addRow);
  }
  el("status").textContent = status;
}

async function showDigest(path, ticket, inside, outside, status) {
  try {
    const data = await get(path);
    if (ticket !== view || !data.groups.length) return;
    el("digest").hidden = false;
    let selection = 0;
    for (const group of data.groups) {
      const button = document.createElement("button");
      button.type = "button";
      button.textContent = group.kind + " · " + group.count;
      button.setAttribute("aria-pressed", "false");
      button.onclick = async () => {
        const current = ++selection;
        if (button.getAttribute("aria-pressed") === "true") {
          button.setAttribute("aria-pressed", "false");
          clearReader();
          showPeriodResults(inside, outside, status);
          return;
        }
        try {
          const selected = await get(path + "&kind=" + encodeURIComponent(group.kind));
          if (ticket !== view || selection !== current) return;
          for (const sibling of el("groups").children) sibling.setAttribute("aria-pressed", String(sibling === button));
          clearReader();
          el("rows").replaceChildren();
          selected.results.forEach(addRow);
          el("status").textContent = "В периоде " + data.total + " писем, показано " + selected.results.length
                                     + " · " + group.kind + " (" + group.count + ")";
        } catch (error) {
          if (ticket === view && current === selection) el("status").textContent = error.message;
        }
      };
      el("groups").append(button);
    }
  } catch (error) {
    if (ticket === view) el("status").textContent += " · Сводка недоступна";
  }
}

async function refresh(append = false) {
  const ticket = ++view;
  el("digest").hidden = true;
  el("digest").open = false;
  el("groups").replaceChildren();
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
    let rows, status, period, digestPath, total, parsedTopic = "";
    if (query) {
      const params = new URLSearchParams({ q: query, mailbox, k: "30" });
      if (el("asof").value) params.set("as_of", el("asof").value);
      const data = await get("search?" + params);
      rows = data.results;
      period = data.period;
      total = data.period_total;
      if (period) {
        const bounds = new URLSearchParams(data.period_bounds);
        if (params.has("as_of")) bounds.set("as_of", params.get("as_of"));
        digestPath = "mailboxes/" + encodeURIComponent(mailbox) + "/digest?" + bounds;
      }
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
      status = "В периоде " + total + " писем, показано " + inside.length;
      showPeriodResults(inside, outside, status);
      showDigest(digestPath, ticket, inside, outside, status);
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
  el("date-hint").textContent = latest ? "Последнее письмо в ящике: " + latest.split("-").reverse().join(".")
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
