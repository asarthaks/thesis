# measure_lengths.py
import torch, pandas as pd, numpy as np, argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from utils import generate

p = argparse.ArgumentParser()
p.add_argument("--model_name", required=True)
p.add_argument("--load_checkpoint_path", default=None)
p.add_argument("--temp", type=float, default=0.9)
p.add_argument("--max_eval_len", type=int, default=25)
p.add_argument("--batch_size", type=int, default=20)
a = p.parse_args()

tok = AutoTokenizer.from_pretrained(a.model_name)
tok.pad_token_id = tok.eos_token_id
m = AutoModelForCausalLM.from_pretrained(a.model_name, torch_dtype=torch.bfloat16).cuda()
if a.load_checkpoint_path:
    m = PeftModel.from_pretrained(m, a.load_checkpoint_path)
m.eval()
eos = tok.eos_token_id

df = pd.read_csv("data/stories.csv")
lengths, samples = [], []
for i in range(100):
    r = df.iloc[i]
    q = f"Beginning: {r.sentence1} {r.sentence2} {r.sentence3}\nEnding: {r.sentence5}\nMiddle:"
    enc = tok(q, return_tensors="pt").input_ids.cuda()
    with torch.no_grad():
        gen, _, _ = generate(m, enc.repeat(a.batch_size,1), eos_token_id=eos,
                             vocab_nice_mask=None, max_len=a.max_eval_len,
                             temperature=a.temp, tokenizer=tok, use_tools=False,
                             limit_capability=None, operators=None, use_cache=True)
    new = gen[:, enc.shape[-1]:]
    for row in new:
        toks = row.tolist()
        n = toks.index(eos) if eos in toks else len(toks)   # tokens before first EOS
        lengths.append(n)
    if i < 5:
        samples.append(tok.decode(new[0], skip_special_tokens=True).strip())

lengths = np.array(lengths)
print(f"mean_len={lengths.mean():.2f}  median={np.median(lengths):.0f}  "
      f"frac_le_5={(lengths<=5).mean():.2f}  frac_le_8={(lengths<=8).mean():.2f}")
print("samples:")
for s in samples: print("  |", s)
