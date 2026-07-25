# G-prime guided-generation examples

Seeded, unfiltered draws. The statistics are in the confusion table (Part 1) and rev_gprime.json; these pairs are illustration only. Verdicts are the guide's (noisy classifier) and the independent judge's (frozen GPT-2 sentiment head). Pairs where guide and judge disagree are the partial instrument alignment made visible, not failures.

Selection rule: np.random.default_rng(0); cells in order [(g2,neg),(g2,pos),(g4,neg),(g4,pos)]; sorted(rng.choice(300,3,replace=False)) per cell; unfiltered

Drawn indices: {"gamma2.0_label0": [153, 190, 253], "gamma2.0_label1": [4, 12, 22], "gamma4.0_label0": [151, 193, 272], "gamma4.0_label1": [163, 189, 217]}


## Cell: gamma 2, target negative

### prompt_idx 153
- **Prompt:** bleakly funny, its characters all the more
- **Unguided:** bleakly funny, its characters all the more often point to negation of value: will you have a sense of humor? or will you have
  - guide: positive | judge: negative (DISAGREE) | span NLL 6.57
- **Guided (gamma 2, toward negative):** bleakly funny, its characters all the more so, their than the the- a and b are the moreso. characters than- the the
  - guide: positive | judge: positive (agree) | span NLL 7.92

### prompt_idx 190
- **Prompt:** all that's missing is the spontaneity
- **Unguided:** all that's missing is the spontaneity, and the courage to pick up undercooked stock,and also the fierce muscle of expectation that will
  - guide: positive | judge: negative (DISAGREE) | span NLL 7.10
- **Guided (gamma 2, toward negative):** all that's missing is the spontaneity. When they leave the missing is the spontaneity. When they leave doing what they are supposed
  - guide: negative | judge: negative (agree) | span NLL 4.44

### prompt_idx 253
- **Prompt:** the quality of the art combined with the humor and
- **Unguided:** the quality of the art combined with the humor and character of the characters proves that the game is good.In my opinion, I want to answer you
  - guide: positive | judge: positive (agree) | span NLL 4.84
- **Guided (gamma 2, toward negative):** the quality of the art combined with the humor and simplicity of the art create a very that is is out of its own right, the arethe and
  - guide: positive | judge: positive (agree) | span NLL 6.52


## Cell: gamma 2, target positive

### prompt_idx 4
- **Prompt:** you don't have to know about music to
- **Unguided:** you don't have to know about music to be musical-- even if you don't know about music  Audio  1 11 1
  - guide: negative | judge: positive (DISAGREE) | span NLL 5.36
- **Guided (gamma 2, toward positive):** you don't have to know about music to play music you t if you don't listen to music about music music y y y n s y
  - guide: positive | judge: negative (DISAGREE) | span NLL 7.10

### prompt_idx 12
- **Prompt:** the emotions are raw and will strike a nerve with
- **Unguided:** the emotions are raw and will strike a nerve with a few steps across the dark for the firstthe bell goes ringing and this quote will be pure in
  - guide: positive | judge: positive (agree) | span NLL 9.96
- **Guided (gamma 2, toward positive):** the emotions are raw and will strike a nerve withthe-number-of-people-who-want-to-make-their-better-
  - guide: positive | judge: positive (agree) | span NLL 5.12

### prompt_idx 22
- **Prompt:** it's a cookie-cutter movie,
- **Unguided:** it's a cookie-cutter movie, same that's where the movies come out, the same majority of them come out of the same industries
  - guide: negative | judge: negative (agree) | span NLL 6.38
- **Guided (gamma 2, toward positive):** it's a cookie-cutter movie, hitit,s a cookie- cut -ter movie, hitit,s
  - guide: negative | judge: negative (agree) | span NLL 6.22


## Cell: gamma 4, target negative

### prompt_idx 151
- **Prompt:** determined to be fun, and bouncy,
- **Unguided:** determined to be fun, and bouncy, and quirky, and exciting, and colourful, and play-determined to be fun-determined
  - guide: positive | judge: positive (agree) | span NLL 4.57
- **Guided (gamma 4, toward negative):** determined to be fun, and bouncy, didn� wasd that to to to and havedd andd, wasd of to,
  - guide: positive | judge: positive (agree) | span NLL 10.83

### prompt_idx 193
- **Prompt:** a psychological thriller with a genuinely spooky premise and
- **Unguided:** a psychological thriller with a genuinely spooky premise and it's only since afterhigh" that he's in jail for full house SH NOT FOR SUPIT
  - guide: positive | judge: positive (agree) | span NLL 9.64
- **Guided (gamma 4, toward negative):** a psychological thriller with a genuinely spooky premise and a overall sort of the to the the� �s n’s done” with
  - guide: positive | judge: positive (agree) | span NLL 10.71

### prompt_idx 272
- **Prompt:** the best film about baseball to hit theaters since field
- **Unguided:** the best film about baseball to hit theaters since fieldy Chavez. This was a game fieldy Chavez. Not bad fieldy Chavez. Made the cut
  - guide: negative | judge: positive (DISAGREE) | span NLL 5.92
- **Guided (gamma 4, toward negative):** the best film about baseball to hit theaters since fielders are athletes were his for his's Baseball  baseball to baseball, be baseball baseball, the the
  - guide: negative | judge: positive (DISAGREE) | span NLL 8.59


## Cell: gamma 4, target positive

### prompt_idx 163
- **Prompt:** pacino is brilliant as the sleep-deprived
- **Unguided:** pacino is brilliant as the sleep-deprived - is adding already of a sleep-deprived (cowned) bristol is brilliant as is
  - guide: negative | judge: positive (DISAGREE) | span NLL 8.40
- **Guided (gamma 4, toward positive):** pacino is brilliant as the sleep-deprived-pacino is the sleep-deprivedpacpacpacpacino is brilliant as the -
  - guide: positive | judge: negative (DISAGREE) | span NLL 8.92

### prompt_idx 189
- **Prompt:** once ( kim ) begins to overplay the
- **Unguided:** once ( kim ) begins to overplay the BILLJOUSE and, others, plays it entirely. is it every ( kim )
  - guide: negative | judge: negative (agree) | span NLL 7.05
- **Guided (gamma 4, toward positive):** once ( kim ) begins to overplay the significance of In that regard, it’s important away see the first the� the interested in
  - guide: positive | judge: positive (agree) | span NLL 10.23

### prompt_idx 217
- **Prompt:** stealing harvard is evidence that the farrell
- **Unguided:** stealing harvard is evidence that the farrell men have not been subject to pressure and blackmail for decades.watch this official and loyalists deceitful
  - guide: negative | judge: positive (DISAGREE) | span NLL 7.85
- **Guided (gamma 4, toward positive):** stealing harvard is evidence that the farrells are from. of.rell far. has stolen from from that the is the farrell far
  - guide: negative | judge: negative (agree) | span NLL 8.08


## Trust-region before/after: Phase 4 off-domain, gamma 4 (the fluency climb the trust region removes)

These are the three highest-NLL guided continuations from the Phase 4 off-domain run (MuCoLa continuation prompts, no trust region). Guidance there pushed text off the fluent manifold (guided NLL ~11 vs unguided ~7). The on-domain gamma-4 G-prime pairs above keep NLL bounded by construction, which is the trust region working.

### off-domain prompt_idx 1, sample 11, target positive
- **Prompt (shared opening):** The book
- **Unguided:** The book was sold!The book was never sold.The book will also never be sold!Somethingうと
  - guide: negative | judge: negative (agree) | span NLL 7.12
- **Guided (gamma 4, no trust region):** The book traded tended end preview store where 'in minorities posed failing Photo minutes united Koreaorean media 4MattIn
  - guide: negative | judge: positive (DISAGREE) | span NLL 16.01

### off-domain prompt_idx 3, sample 14, target positive
- **Prompt (shared opening):** The city
- **Unguided:** The city of provides Toronto I walk nowhere near as fast as I can walk Hyfening speed is a bad
  - guide: negative | judge: negative (agree) | span NLL 8.30
- **Guided (gamma 4, no trust region):** The city proudly absorb 67 loss unemployment mixed off “bah thousand Baronety mile flat mono seaw dialectically larger
  - guide: negative | judge: negative (agree) | span NLL 15.97

### off-domain prompt_idx 11, sample 5, target positive
- **Prompt (shared opening):** The potato
- **Unguided:** The potato located in The House of Commons's foodbox.Smart, well-organized, food-arrant
  - guide: positive | judge: negative (DISAGREE) | span NLL 7.27
- **Guided (gamma 4, no trust region):** The potato problem replied strongly ugly likeThe Tomckswicking meant liter commenting his next early Manhattan critic Angelagi
  - guide: negative | judge: negative (agree) | span NLL 15.92
