import fs from 'fs'
import path from 'path'

const INTRO = `\nACMEModels Repository\nIntroductory remarks: This file is part of the ACMEModels frontend.\n`;

function bannerForFile() {
  return `/*\n${INTRO}*/\n\n`;
}

function hasTopBanner(content) {
  return content.trimStart().startsWith('/*') && content.includes('ACMEModels Repository');
}

function addBanner(content) {
  if (hasTopBanner(content)) return content;
  // Preserve shebang if ever present
  if (content.startsWith('#!')) {
    const idx = content.indexOf('\n');
    return content.slice(0, idx + 1) + bannerForFile() + content.slice(idx + 1);
  }
  return bannerForFile() + content;
}

function addJsDoc(content) {
  // Very lightweight: add JSDoc above function declarations and exported functions if missing
  const lines = content.split(/\n/);
  const out = [];
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    const funcDecl = line.match(/^\s*function\s+(\w+)\s*\(/);
    const exportConst = line.match(/^\s*export\s+const\s+(\w+)\s*=\s*\(/);
    const constFn = line.match(/^\s*const\s+(\w+)\s*=\s*\(/);
    const isDocPrev = out.length && out[out.length - 1].trim().startsWith('/**');
    const addDoc = (name) => {
      if (!isDocPrev) {
        out.push('/**');
        out.push(` * ${name}: Function description.`);
        out.push(' * @param {...any} args');
        out.push(' * @returns {any}');
        out.push(' */');
      }
    };
    if (funcDecl) {
      addDoc(funcDecl[1]);
    } else if (exportConst) {
      addDoc(exportConst[1]);
    } else if (constFn) {
      addDoc(constFn[1]);
    }
    out.push(line);
  }
  return out.join('\n');
}

function processFile(filePath, apply) {
  const content = fs.readFileSync(filePath, 'utf8');
  let updated = addBanner(content);
  updated = addJsDoc(updated);
  const changed = updated !== content;
  if (changed && apply) fs.writeFileSync(filePath, updated, 'utf8');
  return changed;
}

function walk(dir, files = []) {
  for (const entry of fs.readdirSync(dir)) {
    const full = path.join(dir, entry);
    const stat = fs.statSync(full);
    if (stat.isDirectory()) {
      walk(full, files);
    } else if (full.endsWith('.js')) {
      files.push(full);
    }
  }
  return files;
}

const args = process.argv.slice(2);
const apply = args.includes('--apply');
const rootArg = args.includes('--root') ? args[args.indexOf('--root') + 1] : process.cwd();
const files = walk(rootArg);
const changed = [];
for (const f of files) {
  if (processFile(f, apply)) changed.push(f);
}
console.log(`${apply ? 'Modified' : 'Would modify'} ${changed.length} JS files.`);
for (const f of changed) console.log(f);
