(() => {
  'use strict';
  const icons = {
    code: '<path d="m8 7-5 5 5 5m8-10 5 5-5 5m-3-14-2 18"/>',
    model: '<path d="m12 2 9 5v10l-9 5-9-5V7zM3 7l9 5 9-5M12 12v10"/><path d="m7.5 4.5 9 5v5"/>',
    play: '<path d="m9 5 11 7-11 7z"/>',
    gallery: '<rect x="3" y="3" width="7" height="7" rx="1"/><rect x="14" y="3" width="7" height="7" rx="1"/><rect x="3" y="14" width="7" height="7" rx="1"/><rect x="14" y="14" width="7" height="7" rx="1"/>',
    graph: '<rect x="8" y="2" width="8" height="6" rx="2"/><rect x="2" y="16" width="7" height="6" rx="2"/><rect x="15" y="16" width="7" height="6" rx="2"/><path d="M12 8v4H5.5v4M12 12h6.5v4"/>',
    chart: '<path d="M3 3v18h18M7 17V9m5 8V5m5 12v-6"/>',
    limits: '<path d="m12 3 10 18H2zM12 9v5m0 3v1"/>',
    poster: '<rect x="5" y="2" width="14" height="20" rx="2"/><path d="M8 6h8M8 9h5M8 17l3-5 5 5M8 19h8"/>',
    dashboard: '<rect x="2" y="3" width="20" height="18" rx="2"/><path d="M2 8h20M8 8v13m4-5v-4m4 4v-6m3 6v-3"/>',
    slides: '<rect x="2" y="3" width="20" height="14" rx="2"/><path d="M12 17v5m-5 0h10M6 8h5m-5 4h12"/>',
    web: '<rect x="2" y="3" width="20" height="18" rx="2"/><path d="M2 8h20M6 5.5h.1M9 5.5h.1M12 5.5h.1M6 12h5v5H6zm9 0h3m-3 4h3"/>',
    verify: '<path d="m12 2 9 4v6c0 5-9 10-9 10S3 17 3 12V6z"/><path d="m8 12 3 3 5-6"/>',
    balance: '<path d="M12 3v18M4 6h16M3 14l3-8 3 8zm12 0 3-8 3 8zM8 21h8"/>',
    detail: '<circle cx="10" cy="10" r="7"/><path d="m15 15 7 7M7 10h6m-3-3v6"/>',
    palette: '<path d="M12 2a10 10 0 1 0 0 20h1a3 3 0 0 0 2-5c-2-2 1-3 3-3h1c4 0 4-12-7-12z"/><circle cx="7" cy="9" r="1"/><circle cx="12" cy="6" r="1"/><circle cx="17" cy="8" r="1"/>'
  };
  for (const element of document.querySelectorAll('[data-icon]')) {
    element.innerHTML = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" focusable="false" aria-hidden="true">${icons[element.dataset.icon] || icons.code}</svg>`;
  }
  const ns = 'http://www.w3.org/2000/svg';
  const label = (x, y, text, cls = 'diagram-text') => `<text x="${x}" y="${y}" class="${cls}">${text}</text>`;
  const box = (x, y, width, height, cls = 'diagram-box', radius = 12) => `<rect x="${x}" y="${y}" width="${width}" height="${height}" rx="${radius}" class="${cls}"/>`;
  const line = (x1, y1, x2, y2, cls = 'diagram-connector') => `<path d="M${x1} ${y1}L${x2} ${y2}" class="${cls}"/>`;
  const node = (x, y, name, type) => `${box(x,y,110,52,`diagram-node ${type}`)}${label(x+55,y+31,name,'diagram-node-label')}`;
  const svg = (body, name) => `<svg xmlns="${ns}" class="explainer-svg" viewBox="0 0 960 370" role="img" aria-label="${name}"><defs><marker id="flow-arrow" markerWidth="8" markerHeight="8" refX="6" refY="4" orient="auto"><path d="M0 0 8 4 0 8" fill="none" stroke="#729264" stroke-width="1.4"/></marker><pattern id="graph-dots" width="18" height="18" patternUnits="userSpaceOnUse"><circle cx="1" cy="1" r=".8" fill="#a8bea0" opacity=".3"/></pattern></defs><rect width="960" height="370" fill="url(#graph-dots)"/>${body}</svg>`;

  function graphDiagram() {
    const bars = [58, 84, 45, 103, 70].map((height, i) => `<rect x="${75+i*27}" y="${272-height}" width="17" height="${height}" rx="3" class="mini-bar" style="--delay:${i*70}ms"/>`).join('');
    return svg(`${box(35,22,252,318,'artifact-paper')}${label(55,47,'ILLUSTRATIVE LAYOUT','diagram-mini')}${box(55,61,212,61,'design-title')}${label(70,84,'A page,','poster-title')}${label(70,108,'made of relationships.','poster-subtitle')}${box(55,140,212,148,'design-chart')}${line(71,273,248,273,'diagram-axis')}${bars}${box(55,302,96,19,'design-card')}${box(164,302,103,19,'design-card')}
      ${line(303,184,393,184,'diagram-flow')}${label(348,163,'STRUCTURE','diagram-tag')}
      ${line(609,99,609,146)}<path d="M477 194V146H748V194" class="diagram-connector"/>${line(609,146,609,194)}
      ${node(554,47,'Page','node-dark')}${node(422,194,'Title','node-peach')}${node(554,194,'Chart','node-sage')}${node(686,194,'Cards','node-cream')}
      ${label(477,273,'exact text','diagram-node-detail')}${label(609,273,'data + bounds','diagram-node-detail')}${label(741,273,'spacing','diagram-node-detail')}
      ${line(530,220,548,220,'diagram-relation')}${line(664,220,680,220,'diagram-relation')}
      ${label(609,323,'Nodes + attributes + relationships','diagram-bottom')}`, 'Illustration: a page with a title, chart and cards becomes a connected design graph');
  }
  function verifierDiagram() {
    const rules = [['Execution','code'],['Text','poster'],['Boundaries','detail'],['Table / Chart','chart'],['Layout','dashboard'],['Whitespace','slides']];
    const cells = rules.map(([name, icon], i) => {
      const x=340+(i%3)*143, y=71+Math.floor(i/3)*72;
      return `${box(x,y,130,61,'verifier-cell')}<g transform="translate(${x+11} ${y+11}) scale(.8)" fill="none" stroke="#35654b" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">${icons[icon]}</g>${label(x+12,y+47,name,'diagram-cell-label')}`;
    }).join('');
    return svg(`${box(28,113,159,124,'node-dark')}${label(108,146,'DESIGN GRAPH','diagram-white-small')}<g transform="translate(88 165) scale(1.8)" fill="none" stroke="#c8daba" stroke-width="1.3">${icons.graph}</g>
      <path d="M187 175H248V115H319M248 175V282H319" class="diagram-flow"/>
      ${label(341,48,'01 / DETERMINISTIC VERIFIERS','diagram-kicker')}${cells}
      ${box(340,242,416,84,'visual-verifier')}<g transform="translate(356 260) scale(1.5)" fill="none" stroke="#8e513c" stroke-width="1.5">${icons.verify}</g>${label(409,269,'Visual Graph Rubric','diagram-strong')}${label(409,293,'Graph-bound questions → visual judge','diagram-cell-label')}
      <path d="M769 115H810V185H869M769 282H810V185" class="diagram-flow"/>${box(873,159,55,55,'node-dark')}<path d="m888 186 8 8 16-19" fill="none" stroke="#dbeaca" stroke-width="3" stroke-linecap="round"/>
      ${label(480,358,'Different properties. Complementary feedback.','diagram-bottom')}`, 'The design graph routes six properties to deterministic verifiers and visual properties to the Visual Graph Rubric');
  }
  function gdpoDiagram() {
    const colors=['#548269','#8eaa70','#bd785b','#6e8c95'];
    const before=[[12,53,21,66],[72,56,42,65],[8,14,36,22],[25,44,72,16]];
    const after=[[22,55,32,66],[66,42,19,53],[16,28,66,44],[25,46,69,17]];
    const channels=['Text','Data','Layout','Design'];
    const rows=(values,start,normalized)=>values.map((heights,i)=>{
      const x=start+i*73;
      return heights.map((height,j)=>`<rect x="${x+j*13}" y="${247-height}" width="9" height="${height}" rx="2" fill="${colors[i]}" class="reward-column" style="--delay:${i*90+j*35}ms"/>`).join('')+label(x+25,275,channels[i],'diagram-channel')+(normalized?line(x-1,210,x+52,210,'normalization-line'):'');
    }).join('');
    return svg(`${box(24,55,324,253,'artifact-paper')}${label(186,88,'RAW REWARD CHANNELS','diagram-kicker-center')}${label(186,113,'Different scales and densities','diagram-centered-note')}${rows(before,48,false)}${line(48,248,320,248,'diagram-axis')}
      ${line(362,181,425,181,'diagram-flow')}${label(394,153,'GDPO','diagram-tag')}
      ${box(443,55,324,253,'normalized-paper')}${label(605,88,'NORMALIZE EACH CHANNEL','diagram-kicker-center')}${label(605,113,'Within its rollout group','diagram-centered-note')}${rows(after,466,true)}${line(466,248,740,248,'diagram-axis')}
      ${line(784,181,834,181,'diagram-flow')}${box(845,142,90,79,'node-dark')}${label(890,174,'POLICY','diagram-white-small')}${label(890,194,'UPDATE','diagram-white-small')}
      <path d="M890 232V336H186V317" class="diagram-feedback"/>${label(529,360,'Roll out → verify → normalize → update','diagram-bottom')}`, 'Illustrative reward distributions normalized per channel before a policy update; bar heights are schematic, not measured results');
  }
  const steps=[
    {title:'From pixels to properties.', kicker:'DESIGN GRAPH', text:'Represent the page as elements, attributes, and relationships.', draw:graphDiagram},
    {title:'Route each question to the right judge.', kicker:'DECOUPLED REWARDS', text:'Code checks correctness; graph-bound questions guide visual feedback.', draw:verifierDiagram},
    {title:'Balance the signals. Update the policy.', kicker:'GDPO', text:'Normalize each reward channel within its rollout group before combining.', draw:gdpoDiagram}
  ];
  const tabs=[...document.querySelectorAll('.method-tab')];
  function select(index,focus=false) {
    const step=steps[index];
    tabs.forEach((tab,i)=>{tab.classList.toggle('active',i===index);tab.setAttribute('aria-selected',String(i===index));tab.tabIndex=i===index?0:-1;if(focus&&i===index)tab.focus();});
    document.querySelector('#method-panel').setAttribute('aria-labelledby',`step-tab-${index}`);
    document.querySelector('#method-kicker').textContent=step.kicker;
    document.querySelector('#method-panel-title').textContent=step.title;
    document.querySelector('#method-panel-description').textContent=step.text;
    // SVG is fixed explanatory artwork, never untrusted generated markup.
    document.querySelector('#method-diagram').innerHTML=step.draw();
  }
  tabs.forEach((tab,index)=>{
    tab.addEventListener('click',()=>select(index));
    tab.addEventListener('keydown',event=>{
      const direction={ArrowRight:1,ArrowDown:1,ArrowLeft:-1,ArrowUp:-1};
      if(event.key in direction){event.preventDefault();select((index+direction[event.key]+3)%3,true);}
      else if(event.key==='Home'||event.key==='End'){event.preventDefault();select(event.key==='Home'?0:2,true);}
    });
  });
  select(0);
})();
