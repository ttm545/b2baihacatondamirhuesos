const el = id => document.getElementById(id);
let view = 0;
let reading = 0;
let offset = 0;
const pageSize = 30;

async function get(path) {
  const response = await fetch("api/" + path);
  if (!response.ok) throw new Error("Ошибка запроса (" + response.status + ")");
  return response.json();
}

function clearReader(title = "Выберите письмо") {
  ++reading;
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
    el("body").textContent = doc.doc;
    if (matchMedia("(max-width: 780px)").matches) el("subject").scrollIntoView();
  } catch (error) {
    if (ticket === reading) clearReader(error.message);
  }
}

function addRow(doc) {
  const button = document.createElement("button");
  button.className = "mail";
  for (const [tag, value] of [["time", doc.doc_dt || "Без даты"], ["strong", doc.subject], ["span", doc.snippet]]) {
    const field = document.createElement(tag);
    if (tag === "span") field.className = "preview";
    field.textContent = value;
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
    let rows, status, period;
    if (query) {
      const params = new URLSearchParams({ q: query, mailbox, k: "30" });
      if (el("asof").value) params.set("as_of", el("asof").value);
      const data = await get("search?" + params);
      rows = data.results;
      period = data.period;
      status = rows.length ? "Показано: " + rows.length + " · " + data.took_ms + " мс" : "Писем по запросу нет";
    } else {
      rows = await get("mailboxes/" + encodeURIComponent(mailbox) + "/docs?limit=" + pageSize + "&offset=" + offset);
      status = "Показано: " + (offset + rows.length);
    }
    if (ticket !== view) return;
    if (period) {
      el("period").textContent = "Письма за " + period;
      el("period").hidden = false;
    }
    rows.forEach(addRow);
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

el("filters").onsubmit = event => { event.preventDefault(); refresh(); };
el("box").onchange = () => refresh();
el("more").onclick = () => refresh(true);
async function start() {
  try {
    const boxes = await get("mailboxes");
    for (const { mailbox, count } of boxes) {
      const option = new Option(mailbox + " (" + count + ")", mailbox);
      option.dataset.count = count;
      el("box").add(option);
    }
    if (boxes.length) await refresh();
    else el("status").textContent = "В архиве пока нет писем";
  } catch (error) { el("status").textContent = error.message; }
}
start();
