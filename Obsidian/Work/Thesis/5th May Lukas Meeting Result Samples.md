
![[Pasted image 20260505151052.png]]

# CLS Multi Token GPT2 Base


![[Pasted image 20260505150121.png]]
![[Pasted image 20260505150132.png]]

--- Sample 0 | Method: policy | MH: True ---
**Original:** <br>The rostrum of H. americanus bears one or more spines on the underside, which are lacking in H. gammarus.

**Corrupted:** 
The ro orbum of H. americanus bears one or more spines on the underside, which are lacking in H. gam leadersus.

**Predicted:** 
The ro orbum of H. americanus bears one or more spines on the underside, which are lacking in H. gam leadersus.
Acc: 0.0% | KL: 8.3899

--- Sample 0 | Method: policy | MH: False ---
**Original:**  
The rostrum of H. americanus bears one or more spines on the underside, which are lacking in H. gammarus.

<br>**Corrupted:**<br>
The ro orbum of H. americanus bears one or more spines on the underside, which are lacking in H. gam leadersus.

**Predicted:** 
The ro ­um of H. americanus bears one or more spines on the underside, which are lacking in H. gamKEYus.
Acc: 0.0% | KL: 7.4083

--- Sample 0 | Method: grad_norm_preserved_random_dir | MH: True ---
**Original:**  
The rostrum of H. americanus bears one or more spines on the underside, which are lacking in H. gammarus.
<br>**Corrupted:**<br>
The ro orbum of H. americanus bears one or more spines on the underside, which are lacking in H. gam leadersus.

**Predicted:** 
The roigreeum of H. americanus bears one or more spines on the underside, which are lacking in H. gamwormsus.
Acc: 0.0% | KL: 12.4359

--- Sample 0 | Method: grad_norm_preserved_random_dir | MH: False ---
**Original:**  <br>The rostrum of H. americanus bears one or more spines on the underside, which are lacking in H. gammarus.
<br>**Corrupted:**<br> The ro orbum of H. americanus bears one or more spines on the underside, which are lacking in H. gam leadersus.
**Predicted:** The roroomsum of H. americanus bears one or more spines on the underside, which are lacking in H. gamabsorus.
Acc: 0.0% | KL: 12.0807

--- Sample 0 | Method: random | MH: True ---
**Original:**<br>  The rostrum of H. americanus bears one or more spines on the underside, which are lacking in H. gammarus.
<br>**Corrupted:**<br> The ro orbum of H. americanus bears one or more spines on the underside, which are lacking in H. gam leadersus.
<br>**Predicted:**<br> The roaderum of H. americanus bears one or more spines on the underside, which are lacking in H. gamvantageus.
Acc: 0.0% | KL: 9.0318

--- Sample 0 | Method: random | MH: False ---
**Original:**<br>  The rostrum of H. americanus bears one or more spines on the underside, which are lacking in H. gammarus.
<br>**Corrupted:**<br> The ro orbum of H. americanus bears one or more spines on the underside, which are lacking in H. gam leadersus.
<br>**Predicted:**<br> The ro magum of H. americanus bears one or more spines on the underside, which are lacking in H. gamARDus.
Acc: 0.0% | KL: 5.6736

--- Sample 1 | Method: policy | MH: True ---
**Original:**<br>  The spines on the claws of H. americanus are red or red @-@ tipped, while those of H. gammarus are white or white @-@ tipped.
<br>**Corrupted:**<br> The spines on the claws of H. americizons are red or red @-@ tipped, while those of H. gammarus are white or white @- Pit tipped.
<br>**Predicted:**<br> The spines on the claws of H. americizons are red or red @-@ tipped, while those of H. gammarus are white or white @- Pit tipped.
Acc: 0.0% | KL: 6.0750

--- Sample 1 | Method: policy | MH: False ---
**Original:**<br>  The spines on the claws of H. americanus are red or red @-@ tipped, while those of H. gammarus are white or white @-@ tipped.
<br>**Corrupted:**<br> The spines on the claws of H. americizons are red or red @-@ tipped, while those of H. gammarus are white or white @- Pit tipped.
<br>**Predicted:**<br> The spines on the claws of H. americ are red or red @-@ tipped, while those of H. gammarus are white or white @-istani tipped.
Acc: 0.0% | KL: 12.6256

--- Sample 1 | Method: grad_norm_preserved_random_dir | MH: True ---
**Original:**<br>  The spines on the claws of H. americanus are red or red @-@ tipped, while those of H. gammarus are white or white @-@ tipped.
<br>**Corrupted:**<br> The spines on the claws of H. americizons are red or red @-@ tipped, while those of H. gammarus are white or white @- Pit tipped.
<br>**Predicted:**<br> The spines on the claws of H. americasus are red or red @-@ tipped, while those of H. gammarus are white or white @-null tipped.
Acc: 0.0% | KL: 2.0870

--- Sample 1 | Method: grad_norm_preserved_random_dir | MH: False ---
**Original:**<br>  The spines on the claws of H. americanus are red or red @-@ tipped, while those of H. gammarus are white or white @-@ tipped.
<br>**Corrupted:**<br> The spines on the claws of H. americizons are red or red @-@ tipped, while those of H. gammarus are white or white @- Pit tipped.
<br>**Predicted:**<br> The spines on the claws of H. americ Found are red or red @-@ tipped, while those of H. gammarus are white or white @-ito tipped.
Acc: 0.0% | KL: 3.7836

--- Sample 1 | Method: random | MH: True ---
**Original:**<br>  The spines on the claws of H. americanus are red or red @-@ tipped, while those of H. gammarus are white or white @-@ tipped.
<br>**Corrupted:**<br> The spines on the claws of H. americizons are red or red @-@ tipped, while those of H. gammarus are white or white @- Pit tipped.
<br>**Predicted:**<br> The spines on the claws of H. americcow are red or red @-@ tipped, while those of H. gammarus are white or white @-lex tipped.
Acc: 0.0% | KL: 2.7867

--- Sample 1 | Method: random | MH: False ---
**Original:**<br>  The spines on the claws of H. americanus are red or red @-@ tipped, while those of H. gammarus are white or white @-@ tipped.
<br>**Corrupted:**<br> The spines on the claws of H. americizons are red or red @-@ tipped, while those of H. gammarus are white or white @- Pit tipped.
<br>**Predicted:**<br> The spines on the claws of H. americ Kou are red or red @-@ tipped, while those of H. gammarus are white or white @- che tipped.
Acc: 0.0% | KL: 9.3165

# CLS Multi token GPT2 Gflowneft Peft
![[Pasted image 20260505150143.png]]

![[Pasted image 20260505150147.png]]

--- Sample 0 | Method: policy | MH: True ---
**Original:**<br>  The rostrum of H. americanus bears one or more spines on the underside, which are lacking in H. gammarus.
<br>**Corrupted:**<br> The cheersstrum of H. americanus bears one or more spines on the underside, which are lackingfruit H. gammarus.
<br>**Predicted:**<br> The cheersstrum of H. americanus bears one or more spines on the underside, which are lackingfruit H. gammarus.
Acc: 0.0% | KL: 12.5862

--- Sample 0 | Method: policy | MH: False ---
**Original:**<br>  The rostrum of H. americanus bears one or more spines on the underside, which are lacking in H. gammarus.
<br>**Corrupted:**<br> The cheersstrum of H. americanus bears one or more spines on the underside, which are lackingfruit H. gammarus.
<br>**Predicted:**<br> Theretionstrum of H. americanus bears one or more spines on the underside, which are lackingBatman H. gammarus.
Acc: 0.0% | KL: 12.5440

--- Sample 0 | Method: grad_norm_preserved_random_dir | MH: True ---
**Original:**<br>  The rostrum of H. americanus bears one or more spines on the underside, which are lacking in H. gammarus.
<br>**Corrupted:**<br> The cheersstrum of H. americanus bears one or more spines on the underside, which are lackingfruit H. gammarus.
<br>**Predicted:**<br> The Floresstrum of H. americanus bears one or more spines on the underside, which are lackingτ H. gammarus.
Acc: 0.0% | KL: 11.9481

--- Sample 0 | Method: grad_norm_preserved_random_dir | MH: False ---
**Original:**<br>  The rostrum of H. americanus bears one or more spines on the underside, which are lacking in H. gammarus.
<br>**Corrupted:**<br> The cheersstrum of H. americanus bears one or more spines on the underside, which are lackingfruit H. gammarus.
<br>**Predicted:**<br> Theendantstrum of H. americanus bears one or more spines on the underside, which are lacking surges H. gammarus.
Acc: 0.0% | KL: 11.7716

--- Sample 0 | Method: random | MH: True ---
**Original:**<br>  The rostrum of H. americanus bears one or more spines on the underside, which are lacking in H. gammarus.
<br>**Corrupted:**<br> The cheersstrum of H. americanus bears one or more spines on the underside, which are lackingfruit H. gammarus.
<br>**Predicted:**<br> The Dangerstrum of H. americanus bears one or more spines on the underside, which are lacking inhal H. gammarus.
Acc: 0.0% | KL: 13.1312

--- Sample 0 | Method: random | MH: False ---
**Original:**<br>  The rostrum of H. americanus bears one or more spines on the underside, which are lacking in H. gammarus.
<br>**Corrupted:**<br> The cheersstrum of H. americanus bears one or more spines on the underside, which are lackingfruit H. gammarus.
<br>**Predicted:**<br> The eligstrum of H. americanus bears one or more spines on the underside, which are lacking suits H. gammarus.
Acc: 0.0% | KL: 12.4409

--- Sample 1 | Method: policy | MH: True ---
**Original:**<br>  The spines on the claws of H. americanus are red or red @-@ tipped, while those of H. gammarus are white or white @-@ tipped.
<br>**Corrupted:**<br> The spines on the claws of H. **americ** vic are red or red @-@ tipped, while those of H. gammarus are white or white @ precision@ tipped.
<br>**Predicted:**<br> The spines on the claws of H. americ vic are red or red @-@ tipped, while those of H. gammarus are white or white @ precision@ tipped.
Acc: 0.0% | KL: 5.3773

--- Sample 1 | Method: policy | MH: False ---
**Original:**<br>  The spines on the claws of H. americanus are red or red @-@ tipped, while those of H. gammarus are white or white @-@ tipped.
<br>**Corrupted:**<br> The spines on the claws of H. americ vic are red or red @-@ tipped, while those of H. gammarus are white or white @ precision@ tipped.
<br>**Predicted:**<br> The spines on the claws of H. americabase are red or red @-@ tipped, while those of H. gammarus are white or white @ a@ tipped.
Acc: 0.0% | KL: 3.5014

--- Sample 1 | Method: grad_norm_preserved_random_dir | MH: True ---
Original:**  The spines on the claws of H. americanus are red or red @-@ tipped, while those of H. gammarus are white or white @-@ tipped.
<br>**Corrupted:**<br> The spines on the claws of H. americ vic are red or red @-@ tipped, while those of H. gammarus are white or white @ precision@ tipped.
<br>**Predicted:**<br> The spines on the claws of H. americ Ness are red or red @-@ tipped, while those of H. gammarus are white or white @enza@ tipped.
Acc: 0.0% | KL: 4.7197

--- Sample 1 | Method: grad_norm_preserved_random_dir | MH: False ---
**Original:**<br>  The spines on the claws of H. americanus are red or red @-@ tipped, while those of H. gammarus are white or white @-@ tipped.
<br>**Corrupted:**<br> The spines on the claws of H. americ vic are red or red @-@ tipped, while those of H. gammarus are white or white @ precision@ tipped.
<br>**Predicted:**<br> The spines on the claws of H. americRW are red or red @-@ tipped, while those of H. gammarus are white or white @android@ tipped.
Acc: 0.0% | KL: 3.8260

--- Sample 1 | Method: random | MH: True ---
**Original:**<br>  The spines on the claws of H. americanus are red or red @-@ tipped, while those of H. gammarus are white or white @-@ tipped.
<br>**Corrupted:**<br> The spines on the claws of H. americ vic are red or red @-@ tipped, while those of H. gammarus are white or white @ precision@ tipped.
<br>**Predicted:**<br> The spines on the claws of H. americaceous are red or red @-@ tipped, while those of H. gammarus are white or white @Pen@ tipped.
Acc: 0.0% | KL: 4.8597

--- Sample 1 | Method: random | MH: False ---
**Original:**<br>  The spines on the claws of H. americanus are red or red @-@ tipped, while those of H. gammarus are white or white @-@ tipped.
<br>**Corrupted:**<br> The spines on the claws of H. americ vic are red or red @-@ tipped, while those of H. gammarus are white or white @ precision@ tipped.
<br>**Predicted:**<br> The spines on the claws of H. americ eff are red or red @-@ tipped, while those of H. gammarus are white or white @ mot@ tipped.
Acc: 0.0% | KL: 4.0579

# DLS Multi Token GPT2 Base

![[Pasted image 20260505150200.png]]

![[Pasted image 20260505150206.png]]

--- Sample 0 | Method: policy | MH: True ---
**Original:**<br>  The rostrum of H. americanus bears one or more spines on the underside, which are lacking in H. gammarus.
<br>**Corrupted:**<br> The rostrum of Hoult americanus bears one NW more spines on the underside, which are lacking in H. gammarus.
<br>**Predicted:**<br> The rostrum of H GMO americanus bears one 75 more spines on the underside, which are lacking in H. gammarus.
Acc: 0.0% | KL: 8.2030

--- Sample 0 | Method: policy | MH: False ---
**Original:**<br>  The rostrum of H. americanus bears one or more spines on the underside, which are lacking in H. gammarus.
<br>**Corrupted:**<br> The rostrum of Hoult americanus bears one NW more spines on the underside, which are lacking in H. gammarus.
<br>**Predicted:**<br> The rostrum of H362 americanus bears one agricultural more spines on the underside, which are lacking in H. gammarus.
Acc: 0.0% | KL: 7.7381

--- Sample 0 | Method: grad_norm_preserved_random_dir | MH: True ---
**Original:**<br>  The rostrum of H. americanus bears one or more spines on the underside, which are lacking in H. gammarus.
<br>**Corrupted:**<br> The rostrum of Hoult americanus bears one NW more spines on the underside, which are lacking in H. gammarus.
<br>**Predicted:**<br> The rostrum of HAmerican americanus bears one flush more spines on the underside, which are lacking in H. gammarus.
Acc: 0.0% | KL: 7.9977

--- Sample 0 | Method: grad_norm_preserved_random_dir | MH: False ---
**Original:**<br>  The rostrum of H. americanus bears one or more spines on the underside, which are lacking in H. gammarus.
<br>**Corrupted:**<br> The rostrum of Hoult americanus bears one NW more spines on the underside, which are lacking in H. gammarus.
<br>**Predicted:**<br> The rostrum of H Clearly americanus bears oneب more spines on the underside, which are lacking in H. gammarus.
Acc: 0.0% | KL: 6.1717

--- Sample 0 | Method: random | MH: True ---
**Original:**<br>  The rostrum of H. americanus bears one or more spines on the underside, which are lacking in H. gammarus.
<br>**Corrupted:**<br> The rostrum of Hoult americanus bears one NW more spines on the underside, which are lacking in H. gammarus.
<br>**Predicted:**<br> The rostrum of H Same americanus bears one: more spines on the underside, which are lacking in H. gammarus.
Acc: 0.0% | KL: 6.6135

--- Sample 0 | Method: random | MH: False ---
**Original:**<br>  The rostrum of H. americanus bears one or more spines on the underside, which are lacking in H. gammarus.
<br>**Corrupted:**<br> The rostrum of Hoult americanus bears one NW more spines on the underside, which are lacking in H. gammarus.
<br>**Predicted:**<br> The rostrum of HChoose americanus bears one blinding more spines on the underside, which are lacking in H. gammarus.
Acc: 0.0% | KL: 7.5053

--- Sample 1 | Method: policy | MH: True ---
**Original:**<br>  The spines on the claws of H. americanus are red or red @-@ tipped, while those of H. gammarus are white or white @-@ tipped.
<br>**Corrupted:**<br> The spines on the claws of H Believe americanus are red or red @-@ tipped, while those of H. gammarus are white or white @ grotesque@ tipped.
<br>**Predicted:**<br> The spines on the claws of Harma americanus are red or red @-@ tipped, while those of H. gammarus are white or white @ ::@ tipped.
Acc: 0.0% | KL: 5.3970

--- Sample 1 | Method: policy | MH: False ---
**Original:**<br>  The spines on the claws of H. americanus are red or red @-@ tipped, while those of H. gammarus are white or white @-@ tipped.
<br>**Corrupted:**<br> The spines on the claws of H Believe americanus are red or red @-@ tipped, while those of H. gammarus are white or white @ grotesque@ tipped.
<br>**Predicted:**<br> The spines on the claws of H analyzing americanus are red or red @-@ tipped, while those of H. gammarus are white or white @ angry@ tipped.
Acc: 0.0% | KL: 7.2428

--- Sample 1 | Method: grad_norm_preserved_random_dir | MH: True ---
Original:  The spines on the claws of H. americanus are red or red @-@ tipped, while those of H. gammarus are white or white @-@ tipped.
<br>**Corrupted:**<br> The spines on the claws of H Believe americanus are red or red @-@ tipped, while those of H. gammarus are white or white @ grotesque@ tipped.
<br>**Predicted:**<br> The spines on the claws of Hfur americanus are red or red @-@ tipped, while those of H. gammarus are white or white @P@ tipped.
Acc: 0.0% | KL: 5.8407

--- Sample 1 | Method: grad_norm_preserved_random_dir | MH: False ---
Original:  The spines on the claws of H. americanus are red or red @-@ tipped, while those of H. gammarus are white or white @-@ tipped.
<br>**Corrupted:**<br> The spines on the claws of H Believe americanus are red or red @-@ tipped, while those of H. gammarus are white or white @ grotesque@ tipped.
<br>**Predicted:**<br> The spines on the claws of H calcium americanus are red or red @-@ tipped, while those of H. gammarus are white or white @ actively@ tipped.
Acc: 0.0% | KL: 7.2793

--- Sample 1 | Method: random | MH: True ---
Original:  The spines on the claws of H. americanus are red or red @-@ tipped, while those of H. gammarus are white or white @-@ tipped.
<br>**Corrupted:**<br> The spines on the claws of H Believe americanus are red or red @-@ tipped, while those of H. gammarus are white or white @ grotesque@ tipped.
<br>**Predicted:**<br> The spines on the claws of H Extra americanus are red or red @-@ tipped, while those of H. gammarus are white or white @ 2@ tipped.
Acc: 0.0% | KL: 5.8379

--- Sample 1 | Method: random | MH: False ---
Original:  The spines on the claws of H. americanus are red or red @-@ tipped, while those of H. gammarus are white or white @-@ tipped.
<br>**Corrupted:**<br> The spines on the claws of H Believe americanus are red or red @-@ tipped, while those of H. gammarus are white or white @ grotesque@ tipped.
<br>**Predicted:**<br> The spines on the claws of H controlling americanus are red or red @-@ tipped, while those of H. gammarus are white or white @ inexplicable@ tipped.
Acc: 0.0% | KL: 7.1742

# DLS Multi Token GPT2 Gflownet PEFT

![[Pasted image 20260505150229.png]]

![[Pasted image 20260505150232.png]]

--- Sample 0 | Method: policy | MH: True ---
Original:  The rostrum of H. americanus bears one or more spines on the underside, which are lacking in H. gammarus.
<br>**Corrupted:**<br> The unisonstrum of H. americanus bears one or more spines on the underside, which are lacking in H Running gammarus.
<br>**Predicted:**<br> The reflectedstrum of H. americanus bears one or more spines on the underside, which are lacking in Heta gammarus.
Acc: 0.0% | KL: 16.3027

--- Sample 0 | Method: policy | MH: False ---
Original:  The rostrum of H. americanus bears one or more spines on the underside, which are lacking in H. gammarus.
<br>**Corrupted:**<br> The unisonstrum of H. americanus bears one or more spines on the underside, which are lacking in H Running gammarus.
<br>**Predicted:**<br> Thealongstrum of H. americanus bears one or more spines on the underside, which are lacking in Hensual gammarus.
Acc: 0.0% | KL: 13.8071

--- Sample 0 | Method: grad_norm_preserved_random_dir | MH: True ---
Original:  The rostrum of H. americanus bears one or more spines on the underside, which are lacking in H. gammarus.
<br>**Corrupted:**<br> The unisonstrum of H. americanus bears one or more spines on the underside, which are lacking in H Running gammarus.
<br>**Predicted:**<br> The Epstrum of H. americanus bears one or more spines on the underside, which are lacking in H've gammarus.
Acc: 0.0% | KL: 14.0741

--- Sample 0 | Method: grad_norm_preserved_random_dir | MH: False ---
Original:  The rostrum of H. americanus bears one or more spines on the underside, which are lacking in H. gammarus.
<br>**Corrupted:**<br> The unisonstrum of H. americanus bears one or more spines on the underside, which are lacking in H Running gammarus.
<br>**Predicted:**<br> The smashedstrum of H. americanus bears one or more spines on the underside, which are lacking in H out gammarus.
Acc: 0.0% | KL: 15.3773

--- Sample 0 | Method: random | MH: True ---
Original:  The rostrum of H. americanus bears one or more spines on the underside, which are lacking in H. gammarus.
<br>**Corrupted:**<br> The unisonstrum of H. americanus bears one or more spines on the underside, which are lacking in H Running gammarus.
<br>**Predicted:**<br> The holdstrum of H. americanus bears one or more spines on the underside, which are lacking in H67 gammarus.
Acc: 0.0% | KL: 14.0223

--- Sample 0 | Method: random | MH: False ---
Original:  The rostrum of H. americanus bears one or more spines on the underside, which are lacking in H. gammarus.
<br>**Corrupted:**<br> The unisonstrum of H. americanus bears one or more spines on the underside, which are lacking in H Running gammarus.
<br>**Predicted:**<br> The lackedstrum of H. americanus bears one or more spines on the underside, which are lacking in H summoned gammarus.
Acc: 0.0% | KL: 13.6576

--- Sample 1 | Method: policy | MH: True ---
Original:  The spines on the claws of H. americanus are red or red @-@ tipped, while those of H. gammarus are white or white @-@ tipped.
<br>**Corrupted:**<br> The spines on the claws of H. americ ------ are red or red @-@ tipped, HTTP those of H. gammarus are white or white @-@ tipped.
<br>**Predicted:**<br> The spines on the claws of H. americahar are red or red @-@ tipped,press those of H. gammarus are white or white @-@ tipped.
Acc: 0.0% | KL: 2.2552

--- Sample 1 | Method: policy | MH: False ---
Original:  The spines on the claws of H. americanus are red or red @-@ tipped, while those of H. gammarus are white or white @-@ tipped.
<br>**Corrupted:**<br> The spines on the claws of H. americ ------ are red or red @-@ tipped, HTTP those of H. gammarus are white or white @-@ tipped.
<br>**Predicted:**<br> The spines on the claws of H. americ headset are red or red @-@ tipped,Js those of H. gammarus are white or white @-@ tipped.
Acc: 0.0% | KL: 3.4172

--- Sample 1 | Method: grad_norm_preserved_random_dir | MH: True ---
Original:  The spines on the claws of H. americanus are red or red @-@ tipped, while those of H. gammarus are white or white @-@ tipped.
<br>**Corrupted:**<br> The spines on the claws of H. americ ------ are red or red @-@ tipped, HTTP those of H. gammarus are white or white @-@ tipped.
<br>**Predicted:**<br> The spines on the claws of H. americifi are red or red @-@ tipped,.# those of H. gammarus are white or white @-@ tipped.
Acc: 0.0% | KL: 2.6903

--- Sample 1 | Method: grad_norm_preserved_random_dir | MH: False ---
Original:  The spines on the claws of H. americanus are red or red @-@ tipped, while those of H. gammarus are white or white @-@ tipped.
<br>**Corrupted:**<br> The spines on the claws of H. americ ------ are red or red @-@ tipped, HTTP those of H. gammarus are white or white @-@ tipped.
<br>**Predicted:**<br> The spines on the claws of H. americ lovely are red or red @-@ tipped,ellectual those of H. gammarus are white or white @-@ tipped.
Acc: 0.0% | KL: 4.0235

--- Sample 1 | Method: random | MH: True ---
Original:  The spines on the claws of H. americanus are red or red @-@ tipped, while those of H. gammarus are white or white @-@ tipped.
<br>**Corrupted:**<br> The spines on the claws of H. americ ------ are red or red @-@ tipped, HTTP those of H. gammarus are white or white @-@ tipped.
<br>**Predicted:**<br> The spines on the claws of H. americ ancestry are red or red @-@ tipped, Fla those of H. gammarus are white or white @-@ tipped.
Acc: 0.0% | KL: 2.8981

--- Sample 1 | Method: random | MH: False ---
Original:  The spines on the claws of H. americanus are red or red @-@ tipped, while those of H. gammarus are white or white @-@ tipped.
<br>**Corrupted:**<br> The spines on the claws of H. americ ------ are red or red @-@ tipped, HTTP those of H. gammarus are white or white @-@ tipped.
<br>**Predicted:**<br> The spines on the claws of H. americ managed are red or red @-@ tipped, risky those of H. gammarus are white or white @-@ tipped.
Acc: 0.0% | KL: 4.2978