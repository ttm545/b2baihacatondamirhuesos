const decoder = new TextDecoder("utf-8", { fatal: true });
const words = value => value.toLowerCase().replaceAll("ё", "е").match(/[\p{L}\p{N}]+/gu) || [];
const required = ["documents.jsonl", "queries_dev.jsonl", "qrels_dev.jsonl"];

export async function readCorpus(file) {
  if (file.size > 64 * 1024 ** 2) throw new Error("Архив слишком большой для этой страницы.");
  const buffer = await file.arrayBuffer();
  const view = new DataView(buffer);
  let end = view.byteLength - 22;
  const limit = Math.max(0, end - 0xffff);
  while (end >= limit) {
    if (view.getUint32(end, true) === 0x06054b50 && end + 22 + view.getUint16(end + 20, true) === view.byteLength) break;
    end--;
  }
  if (end < limit) throw new Error("Не удалось распознать ZIP-архив.");
  let entry = view.getUint32(end + 16, true);
  const count = view.getUint16(end + 10, true);
  const files = new Map();
  for (let i = 0; i < count; i++) {
    if (entry + 46 > end || view.getUint32(entry, true) !== 0x02014b50) throw new Error("Повреждён список файлов ZIP.");
    const nameLength = view.getUint16(entry + 28, true);
    const nameBytes = buffer.slice(entry + 46, entry + 46 + nameLength);
    const name = decoder.decode(nameBytes).split("/").at(-1);
    if (required.includes(name)) {
      if (files.has(name)) throw new Error("В архиве встречаются одинаковые имена файлов.");
      const method = view.getUint16(entry + 10, true);
      const compressed = view.getUint32(entry + 20, true);
      const size = view.getUint32(entry + 24, true);
      const local = view.getUint32(entry + 42, true);
      if (size > 80 * 1024 ** 2) throw new Error("Распакованный файл слишком большой.");
      if (local + 30 > entry || view.getUint32(local, true) !== 0x04034b50) throw new Error("Повреждён файл в архиве.");
      const start = local + 30 + view.getUint16(local + 26, true) + view.getUint16(local + 28, true);
      if (start + compressed > entry) throw new Error("Повреждён файл в архиве.");
      const blob = new Blob([buffer.slice(start, start + compressed)]);
      const bytes = method === 0 ? await blob.arrayBuffer() : await inflate(blob, method, size);
      if (bytes.byteLength !== size) throw new Error("Размер распакованного файла не совпадает.");
      const lines = decoder.decode(bytes).split(/\r?\n/).filter(line => line.trim());
      files.set(name, lines.map(line => JSON.parse(line)));
    }
    entry += 46 + nameLength + view.getUint16(entry + 30, true) + view.getUint16(entry + 32, true);
  }
  for (const name of required) if (!files.has(name)) throw new Error(`В архиве нет ${name}.`);
  const [documents, queries, judgments] = required.map(name => files.get(name));
  if (!documents.length || !documents.every(d => d.doc_id != null && d.synthetic_user != null && typeof d.doc === "string"
    && (d.doc_dt == null || /^\d{4}-\d{2}-\d{2}$/.test(d.doc_dt)))) throw new Error("Проверьте формат documents.jsonl.");
  if (!queries.length || !queries.every(q => q.query_id != null && q.synthetic_user != null
    && typeof q.query === "string" && /^\d{4}-\d{2}-\d{2}$/.test(q.query_date))) throw new Error("Проверьте формат queries_dev.jsonl.");
  if (!judgments.every(q => q.query_id != null && q.doc_id != null && Number.isFinite(q.relevance))) throw new Error("Проверьте формат qrels_dev.jsonl.");
  return { documents, queries, judgments };
}

async function inflate(blob, method, size) {
  if (method !== 8) throw new Error("Способ сжатия ZIP не поддерживается.");
  const reader = blob.stream().pipeThrough(new DecompressionStream("deflate-raw")).getReader();
  const out = new Uint8Array(size);
  let offset = 0;
  try {
    while (true) {
      const chunk = await reader.read();
      if (chunk.done) break;
      if (offset + chunk.value.length > size) {
        await reader.cancel();
        throw new Error("Данные ZIP превышают заявленный размер.");
      }
      out.set(chunk.value, offset);
      offset += chunk.value.length;
    }
  } finally {
    reader.releaseLock();
  }
  return out.subarray(0, offset);
}

export function createIndex(documents) {
  const boxes = new Map();
  for (const doc of documents) {
    const key = String(doc.synthetic_user);
    if (!boxes.has(key)) boxes.set(key, { docs: [], df: new Map(), length: 0 });
    const box = boxes.get(key);
    const tokens = words(doc.doc);
    const tf = new Map();
    for (const token of tokens) tf.set(token, (tf.get(token) || 0) + 1);
    for (const token of tf.keys()) box.df.set(token, (box.df.get(token) || 0) + 1);
    box.docs.push({ doc, tf, length: tokens.length });
    box.length += tokens.length;
  }
  return {
    mailboxes: [...boxes.keys()].sort((a, b) => a.localeCompare(b, "ru", { numeric: true })),
    search(query, mailbox, asOf = "", limit = 25) {
      const box = boxes.get(String(mailbox));
      if (!box) return [];
      const avg = box.length / box.docs.length || 1;
      const terms = [...new Set(words(query))].map(token => {
        const df = box.df.get(token) || 0;
        return { token, idf: Math.log(1 + (box.docs.length - df + 0.5) / (df + 0.5)) };
      });
      const results = [];
      for (const { doc, tf, length } of box.docs) {
        if (asOf && doc.doc_dt && doc.doc_dt > asOf) continue;
        let score = 0;
        const norm = 1.2 * (0.25 + 0.75 * length / avg);
        for (const { token, idf } of terms) {
          const count = tf.get(token) || 0;
          if (count) score += idf * count * 2.2 / (count + norm);
        }
        if (score || !terms.length) results.push({ doc, score });
      }
      return results.sort((a, b) => b.score - a.score || (b.doc.doc_dt || "").localeCompare(a.doc.doc_dt || ""))
        .slice(0, limit);
    }
  };
}

export async function evaluate(index, queries, judgments, k, onProgress = () => {}) {
  const truth = new Map();
  for (const item of judgments) {
    if (item.relevance <= 0) continue;
    const key = String(item.query_id);
    if (!truth.has(key)) truth.set(key, new Map());
    truth.get(key).set(String(item.doc_id), item.relevance);
  }
  const rows = [];
  let recall = 0, ndcg = 0, mrr = 0, scored = 0;
  const latencies = [];
  for (const query of queries) {
    const start = performance.now();
    const hits = index.search(query.query, query.synthetic_user, query.query_date, k);
    const ms = performance.now() - start;
    latencies.push(ms);
    const relevant = truth.get(String(query.query_id));
    const ids = hits.map(({ doc }) => doc.doc_id);
    let rowRecall = null, rowNdcg = null, rowMrr = null;
    if (relevant?.size) {
      const grades = ids.map(id => relevant.get(String(id)) || 0);
      const dcg = values => values.reduce((sum, grade, i) => sum + (2 ** grade - 1) / Math.log2(i + 2), 0);
      rowRecall = grades.filter(Boolean).length / relevant.size;
      rowNdcg = dcg(grades) / dcg([...relevant.values()].sort((a, b) => b - a).slice(0, k));
      const first = grades.findIndex(Boolean);
      rowMrr = first < 0 ? 0 : 1 / (first + 1);
      recall += rowRecall;
      ndcg += rowNdcg;
      mrr += rowMrr;
      scored++;
    }
    rows.push({ query, ids, recall: rowRecall, ndcg: rowNdcg, mrr: rowMrr });
    if (rows.length % 15 === 0) {
      onProgress(rows.length);
      await new Promise(resolve => setTimeout(resolve, 0));
    }
  }
  latencies.sort((a, b) => a - b);
  const p99 = latencies[Math.ceil(latencies.length * 0.99) - 1] ?? 0;
  return { rows, scored, recall: scored ? recall / scored : null, ndcg: scored ? ndcg / scored : null,
    mrr: scored ? mrr / scored : null, p99 };
}
