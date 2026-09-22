const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict');
const html=fs.readFileSync(__dirname+'/index.html','utf8'),mascot={dataset:{}};
const code=html.slice(html.indexOf('function setMascot('),html.indexOf('function renderEmotion('));
const c={$:()=>mascot};vm.createContext(c);vm.runInContext(code,c);
for(const label of ['calm','happy','neutral','sad','fear','disgust','angry','surprise']){c.setMascot(label);assert.equal(mascot.dataset.expression,label);}
c.setMascot('unexpected');assert.equal(mascot.dataset.expression,'idle');c.setMascot(null);assert.equal(mascot.dataset.expression,'idle');
assert.ok(html.includes("latestEmotion=data;setMascot(data.emotion)"));assert.ok(html.includes("latestEmotion=null;setMascot('idle')"));
console.log('Mascot mapping, update hookup and reset checks passed.');
