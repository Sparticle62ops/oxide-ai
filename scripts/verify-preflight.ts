import { readFile, writeFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
const root = process.argv[2];
if (!root) throw new Error('Usage: bun verify-preflight.ts STUDY_ROOT [RUNNER_BINARY]');
const binaryPath = process.argv[3];
const hash = (s: Buffer | string) => createHash('sha256').update(s).digest('hex');
const json = async (path: string) => JSON.parse(await readFile(path,'utf8'));
const assert = (ok: unknown, detail: string) => { if (!ok) throw new Error(detail); };
const fnv = (text: string) => { let h=0xcbf29ce484222325n; for(const b of Buffer.from(text)) h=BigInt.asUintN(64,(h^BigInt(b))*0x100000001b3n); return h.toString(16).padStart(16,'0'); };
const provenance = await json(`${root}/provenance.json`);
for (const entry of Object.values(provenance.inputs) as {path:string,sha256:string}[]) {
 assert(hash(await readFile(entry.path)) === entry.sha256, `Input SHA-256 changed: ${entry.path}`);
}
if(binaryPath) assert(hash(await readFile(binaryPath))===provenance.build.binary_sha256,'Runner binary differs from recorded accepted build');
const controls: Record<string, number> = {epochs:4,seed:42,chunk:64,accumulate:4,warmup_steps:20,base_lr:0.001,state:8,key:16,memory:32,expected_transitions_per_epoch:840951,checkpoint_every_updates:512,diagnostics_every_updates:256};
const results: any[] = [];
let common: string | undefined;
for(const [width,depth] of [[64,1],[256,1],[64,2],[64,4]]) {
 const name=`w${width}-d${depth}`;
 const manifestPath=`${root}/manifests/${name}-run.json`;
 const bytes=await readFile(manifestPath); const manifest=JSON.parse(bytes.toString());
 for(const [key,value] of Object.entries(controls)) assert(manifest[key]===value,`${name}: frozen ${key} mismatch`);
 assert(manifest.width===width && manifest.depth===depth,`${name}: wrong architecture`);
 assert(!manifest.preflight_only && manifest.max_updates_this_invocation===undefined,`${name}: primary manifest must represent a complete run`);
 const preflight=await json(`${manifest.run_dir}/preflight.json`);
 const frozen=preflight.frozen;
 assert(preflight.status==='preflight-ok',`${name}: preflight failed`);
 assert(preflight.actual_transitions_per_epoch===840951 && frozen.training_plan.epochs===4,`${name}: exposure mismatch`);
 assert(preflight.total_updates===frozen.training_plan.groups_per_epoch*4,`${name}: schedule horizon mismatch`);
 assert(frozen.model.width===width && frozen.model.depth===depth,`${name}: preflight architecture mismatch`);
 for(const [key,value] of Object.entries({beta1:0.9,beta2:0.999,weight_decay:0.01,eps:1e-8,tau_mem:0.1,ema_alpha:0.01,lr:0.001}))
  assert(Math.fround(frozen.model[key])===Math.fround(value),`${name}: optimizer/model ${key} mismatch`);
 const source=frozen.tokenizer_source_model;
 for(const [key,value] of Object.entries({width:64,depth:1,state:8,key:16,memory:32,chunk:64,optimizer_updates:0}))
  assert(source?.[key]===value,`${name}: wrong width64 initial control ${key}`);
 assert(source.checkpoint_format==='V7',`${name}: initial control is not V7`);
 const metadata=preflight.tokenizer_metadata;
 assert(typeof metadata==='string',`${name}: tokenizer metadata missing`);
 assert(Buffer.byteLength(metadata)===frozen.files.tokenizer_metadata.bytes && fnv(metadata)===frozen.files.tokenizer_metadata.fnv1a64,`${name}: exported metadata does not match initial checkpoint metadata`);
 const prompt=await json(manifest.prompts_path);
 assert(JSON.stringify(prompt.sampling.temperatures)==='[0,0.7]' && prompt.sampling.seed===1337 && prompt.sampling.max_new_tokens===64 && prompt.prompts.length===8,`${name}: prompt sampling changed`);
 for(const [kind,field] of [['train','train_path'],['validation','validation_path'],['prompts','prompts_path']] as const) {
  assert(manifest[field]===provenance.inputs[kind].path,`${name}: wrong ${kind} source`);
 }
 assert(manifest.initial_checkpoint===provenance.inputs['tokenizer-control-initial.pssa'].path,`${name}: wrong initial checkpoint source`);
 const comparable=JSON.stringify({files:frozen.files,training_plan:frozen.training_plan,seed:frozen.seed,source,tokenizer:hash(metadata),sampling:frozen.prompt_sampling,checkpoint_every_updates:frozen.checkpoint_every_updates,diagnostics_every_updates:frozen.diagnostics_every_updates});
 if(common===undefined) common=comparable; else assert(common===comparable,`${name}: non-architectural comparison controls differ`);
 results.push({name,width,depth,parameters:preflight.parameter_count,transitions_per_epoch:preflight.actual_transitions_per_epoch,total_transitions:840951*4,total_updates:preflight.total_updates,tokenizer_metadata_sha256:hash(metadata),manifest_sha256:hash(bytes),preflight_sha256:hash(await readFile(`${manifest.run_dir}/preflight.json`))});
}
const report={status:'accepted',verified_at:new Date().toISOString(),checks:'All four exact frozen manifests; SHA-256 input and tokenizer provenance; width64/step0/V7 tokenizer source; matching controls; full four-pass exposure; fixed 16-generation policy; no test-set access.',binary_sha256:provenance.build.binary_sha256,configurations:results};
await writeFile(`${root}/preflight-acceptance.json`,JSON.stringify(report,null,2)+'\n');
console.log(JSON.stringify(report,null,2));
