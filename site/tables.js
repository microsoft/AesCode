(() => {
  'use strict';

  // Strings preserve the published README's precision.
  window.AESCODE_TABLES = [
    {
        "id":  "table-main-results",
        "title":  "Full infographic benchmark results",
        "description":  "300 infographic samples; scores (%) averaged over three generations per prompt.",
        "columns":  [
                        {
                            "key":  "model",
                            "label":  "Model"
                        },
                        {
                            "key":  "reference",
                            "label":  "Ref."
                        },
                        {
                            "key":  "text",
                            "label":  "Text"
                        },
                        {
                            "key":  "bound",
                            "label":  "Bound"
                        },
                        {
                            "key":  "chart",
                            "label":  "Chart"
                        },
                        {
                            "key":  "rule",
                            "label":  "Rule"
                        },
                        {
                            "key":  "content",
                            "label":  "Content"
                        },
                        {
                            "key":  "layout",
                            "label":  "Layout"
                        },
                        {
                            "key":  "style",
                            "label":  "Style"
                        },
                        {
                            "key":  "visual",
                            "label":  "Visual"
                        },
                        {
                            "key":  "overall",
                            "label":  "Overall"
                        }
                    ],
        "rows":  [
                     {
                         "model":  "GPT-5.5",
                         "reference":  "No",
                         "text":  "88.67",
                         "bound":  "86.18",
                         "chart":  "82.85",
                         "rule":  "85.90",
                         "content":  "79.27",
                         "layout":  "68.09",
                         "style":  "52.80",
                         "visual":  "66.72",
                         "overall":  "76.31",
                         "_ours":  false,
                         "_best":  [

                                   ],
                         "_second":  [

                                     ]
                     },
                     {
                         "model":  "GPT-5.5",
                         "reference":  "Yes",
                         "text":  "89.22",
                         "bound":  "86.85",
                         "chart":  "81.35",
                         "rule":  "85.80",
                         "content":  "82.44",
                         "layout":  "89.93",
                         "style":  "57.91",
                         "visual":  "76.76",
                         "overall":  "81.28",
                         "_ours":  false,
                         "_best":  [
                                       "style"
                                   ],
                         "_second":  [
                                         "layout"
                                     ]
                     },
                     {
                         "model":  "Claude Opus 4.8",
                         "reference":  "No",
                         "text":  "91.40",
                         "bound":  "85.03",
                         "chart":  "87.69",
                         "rule":  "88.04",
                         "content":  "84.98",
                         "layout":  "64.88",
                         "style":  "46.72",
                         "visual":  "65.53",
                         "overall":  "76.78",
                         "_ours":  false,
                         "_best":  [

                                   ],
                         "_second":  [

                                     ]
                     },
                     {
                         "model":  "Claude Opus 4.8",
                         "reference":  "Yes",
                         "text":  "92.45",
                         "bound":  "84.43",
                         "chart":  "73.75",
                         "rule":  "83.55",
                         "content":  "86.42",
                         "layout":  "89.76",
                         "style":  "55.51",
                         "visual":  "77.23",
                         "overall":  "80.39",
                         "_ours":  false,
                         "_best":  [
                                       "content"
                                   ],
                         "_second":  [
                                         "visual"
                                     ]
                     },
                     {
                         "model":  "Qwen3-VL-8B",
                         "reference":  "No",
                         "text":  "58.32",
                         "bound":  "77.23",
                         "chart":  "43.30",
                         "rule":  "59.61",
                         "content":  "38.33",
                         "layout":  "23.97",
                         "style":  "13.22",
                         "visual":  "25.17",
                         "overall":  "42.39",
                         "_ours":  false,
                         "_best":  [

                                   ],
                         "_second":  [

                                     ]
                     },
                     {
                         "model":  "Qwen3-VL-8B",
                         "reference":  "Yes",
                         "text":  "64.07",
                         "bound":  "70.99",
                         "chart":  "42.35",
                         "rule":  "59.14",
                         "content":  "52.90",
                         "layout":  "51.33",
                         "style":  "29.92",
                         "visual":  "44.72",
                         "overall":  "51.93",
                         "_ours":  false,
                         "_best":  [

                                   ],
                         "_second":  [

                                     ]
                     },
                     {
                         "model":  "Qwen3-VL-32B",
                         "reference":  "No",
                         "text":  "66.00",
                         "bound":  "80.31",
                         "chart":  "41.36",
                         "rule":  "62.55",
                         "content":  "49.81",
                         "layout":  "34.85",
                         "style":  "19.34",
                         "visual":  "34.67",
                         "overall":  "48.61",
                         "_ours":  false,
                         "_best":  [

                                   ],
                         "_second":  [

                                     ]
                     },
                     {
                         "model":  "Qwen3-VL-32B",
                         "reference":  "Yes",
                         "text":  "71.19",
                         "bound":  "79.61",
                         "chart":  "50.77",
                         "rule":  "67.19",
                         "content":  "62.04",
                         "layout":  "64.59",
                         "style":  "38.43",
                         "visual":  "55.02",
                         "overall":  "61.10",
                         "_ours":  false,
                         "_best":  [

                                   ],
                         "_second":  [

                                     ]
                     },
                     {
                         "model":  "AesCode-8B",
                         "reference":  "Yes",
                         "text":  "94.06",
                         "bound":  "88.36",
                         "chart":  "87.79",
                         "rule":  "90.07",
                         "content":  "86.41",
                         "layout":  "87.80",
                         "style":  "53.21",
                         "visual":  "75.80",
                         "overall":  "82.94",
                         "_ours":  true,
                         "_best":  [

                                   ],
                         "_second":  [
                                         "text",
                                         "bound",
                                         "chart",
                                         "rule",
                                         "content",
                                         "overall"
                                     ]
                     },
                     {
                         "model":  "AesCode-32B",
                         "reference":  "Yes",
                         "text":  "95.34",
                         "bound":  "97.27",
                         "chart":  "90.37",
                         "rule":  "94.33",
                         "content":  "85.76",
                         "layout":  "90.58",
                         "style":  "55.99",
                         "visual":  "77.44",
                         "overall":  "85.89",
                         "_ours":  true,
                         "_best":  [
                                       "text",
                                       "bound",
                                       "chart",
                                       "rule",
                                       "layout",
                                       "visual",
                                       "overall"
                                   ],
                         "_second":  [
                                         "style"
                                     ]
                     }
                 ],
        "note":  "Ref. indicates image-reference input. Rule averages Text, Bound, and Chart; Visual averages Content, Layout, and Style. Overall averages Rule and Visual. Higher is better. Bold: best; underline: second best.",
        "sourceLabel":  "Published README · Main comparison"
    }
];

  const element = (tag, className, text) => {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  };

  function renderTable(data) {
    const section = element('section', 'paper-table-section');
    section.id = data.id;
    const heading = element('h3', 'paper-table-title', data.title);
    heading.id = `${data.id}-title`;
    section.setAttribute('aria-labelledby', heading.id);
    section.append(element('p', 'paper-table-source', data.sourceLabel), heading);
    const description = element('p', 'paper-table-description', data.description);
    description.id = `${data.id}-description`;
    const hint = element('p', 'paper-table-hint', 'All rows shown. Scroll horizontally to inspect every column.');
    hint.id = `${data.id}-hint`;
    const note = element('p', 'paper-table-note', data.note);
    note.id = `${data.id}-note`;
    section.append(description, hint);
    const scroller = element('div', 'paper-table-scroll');
    scroller.tabIndex = 0;
    scroller.setAttribute('role', 'region');
    scroller.setAttribute('aria-labelledby', heading.id);
    scroller.setAttribute('aria-describedby', hint.id);
    const table = element('table', 'paper-research-table');
    table.setAttribute('aria-describedby', `${description.id} ${note.id}`);
    table.append(element('caption', 'paper-table-sr-only', `${data.sourceLabel}: ${data.title}`));
    const head = element('thead');
    const headerRow = element('tr');
    for (const column of data.columns) {
      const th = element('th', '', column.label);
      th.scope = 'col';
      th.id = `${data.id}-column-${column.key}`;
      headerRow.append(th);
    }
    head.append(headerRow);
    const body = element('tbody');
    data.rows.forEach((row, rowIndex) => {
      const tr = element('tr', row._ours ? 'paper-ours-row' : '');
      const rowId = `${data.id}-row-${rowIndex}`;
      for (const column of data.columns) {
        const isModel = column.key === 'model';
        const cell = element(isModel ? 'th' : 'td', isModel ? 'paper-row-label' : '');
        if (isModel) {
          cell.scope = 'row';
          cell.id = rowId;
        }
        cell.setAttribute('headers', `${data.id}-column-${column.key}${isModel ? '' : ` ${rowId}`}`);
        if (['rule', 'visual', 'overall'].includes(column.key)) cell.classList.add(`paper-metric-${column.key}`);
        const best = row._best.includes(column.key);
        const second = row._second.includes(column.key);
        const bold = best || (row._ours && ['model', 'reference'].includes(column.key));
        const value = element(bold ? 'strong' : 'span', second ? 'paper-second' : '', row[column.key]);
        if (best || second) value.title = best ? 'Best published score' : 'Second-best published score';
        cell.append(value);
        tr.append(cell);
      }
      body.append(tr);
    });
    table.append(head, body);
    scroller.append(table);
    section.append(scroller, note);
    return section;
  }

  function render() {
    const host = document.getElementById('paper-tables');
    if (!host || host.dataset.tablesRendered === 'true') return;
    host.replaceChildren(renderTable(window.AESCODE_TABLES[0]));
    host.dataset.tablesRendered = 'true';
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', render, {once: true});
  else render();
})();
