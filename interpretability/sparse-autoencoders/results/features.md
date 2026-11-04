# A random sample of 30 live latents (topk SAE, k = 32)

Each row: the latent's firing rate on held-out text, the most common current byte among its 20 strongest activations (with purity), the most common next byte, and its three strongest contexts (the activating byte in brackets; ⏎ = newline, ␣ = space).

| Latent | Fires on | Current byte (purity) | Next byte (purity) | Top contexts |
|---|---|---|---|---|
| 5 | 9.47 % | ␣ (95 %) | ␣ (100 %) | `andle_queue.pop(0)⏎␣␣␣␣␣[␣]␣␣␣␣`<br>`sult,␣exc_info[1],⏎␣␣␣␣␣[␣]␣␣␣␣`<br>`f._filelineno␣+=␣1⏎␣␣␣␣␣[␣]␣␣␣␣` |
| 35 | 0.72 % | p (100 %) | e (85 %) | `[p]uted`<br>`[p]t␣In`<br>`[p]eat(` |
| 70 | 5.04 % | u (95 %) | 0 (55 %) | `⏎␣␣␣␣'\[u]0167`<br>`-THAN␣OR␣EQUAL␣TO⏎␣␣␣␣'\[u]2264`<br>`b0␣->␣LIGHT␣SHADE⏎␣␣␣␣'\[u]2592` |
| 84 | 1.76 % | d (45 %) | d (55 %) | `␣␣␣␣action.default␣=␣kwa[r]gs[a`<br>`␣␣category=DeprecationWa[r]ning`<br>`␣␣␣if␣action.dest␣in␣kwa[r]gs:⏎` |
| 154 | 0.49 % | r (80 %) | ␣ (20 %) | `lict_handle[r])⏎⏎␣`<br>`t␣(c)␣2006-2008,␣R␣Oudke[r]k⏎#␣`<br>`acker:␣WindowsLoadTracke[r]␣\|␣N` |
| 359 | 0.41 % | s (80 %) | e (50 %) | `␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣result.__[s]elf_`<br>`._last_accepted␣=␣self._[s]ocke`<br>`␣ov␣=␣self._[s]end_` |
| 552 | 0.67 % | a (100 %) | s (55 %) | `␣server␣generate␣a␣new␣p[a]sswo`<br>`␣The␣user_function␣was␣p[a]ssed`<br>`upplied,␣is␣passed␣to␣'p[a]rse'` |
| 572 | 0.06 % | ␣ (35 %) | ␣ (45 %) | `⏎␣␣␣␣)⏎\x00"["]"Hel`<br>`0'␣␣␣#␣␣0xE4␣->␣PER␣MILL[E]␣SIG`<br>`␣␣␣#␣␣0x00f3␣->␣LESS-THA[N]␣OR␣` |
| 627 | 1.49 % | e (70 %) | ␣ (40 %) | `ine⏎␣␣␣␣can␣be␣a␣callabl[e]␣fun`<br>`SONDecod[e]r(ob`<br>`␣␣␣␣␣␣␣␣␣raise␣JSONDecod[e]Erro` |
| 815 | 0.29 % | n (100 %) | f (35 %) | `␣''⏎␣␣␣␣␣␣␣␣␣␣␣␣data['co[n]firm`<br>`cking(True)⏎␣␣␣␣␣␣␣␣s.co[n]nect`<br>`guments␣to␣the␣Action␣co[n]stru` |
| 1027 | 0.45 % | ␣ (75 %) | a (20 %) | `redi[c]ate␣`<br>`t_handler␣--␣String␣indi[c]atin`<br>`args",␣"keywords",␣"__di[c]t__"` |
| 1037 | 0.36 % | N (100 %) | G (85 %) | `AWI[N]GS␣D`<br>`OX␣DRAWI[N]GS␣L`<br>`0x2554,␣␣␣␣␣#␣␣BOX␣DRAWI[N]GS␣D` |
| 1112 | 0.47 % | a (35 %) | l (20 %) | `nto␣the␣resulting␣MRO.␣U[n]rela`<br>`u[n]less`<br>`U[A]L␣TO` |
| 1138 | 0.20 % | E (40 %) | r (35 %) | `ARK⏎␣␣␣␣'\u201e'␣␣␣#␣␣0x[E]3␣->`<br>`OVE⏎␣␣␣␣'\xe6'␣␣␣␣␣#␣␣0x[E]6␣->`<br>`IGN⏎␣␣␣␣'\xc2'␣␣␣␣␣#␣␣0x[E]5␣->` |
| 1144 | 0.08 % | E (65 %) | ␣ (70 %) | `␣DRAWINGS␣VERTICAL␣SINGL[E]␣AND`<br>`->␣BOX␣DRAWINGS␣UP␣SINGL[E]␣AND`<br>`#␣␣BOX␣DRAWINGS␣UP␣SINGL[E]␣AND` |
| 1236 | 2.01 % | e (95 %) | r (25 %) | `er(object):⏎␣␣␣␣"""Simpl[e]␣JSO`<br>`Reader(Codec,␣mbc.Multib[y]teSt`<br>`pes*␣which␣are␣also␣impl[e]ment` |
| 1288 | 0.31 % | f (35 %) | d (40 %) | `roup[,]␣sel`<br>`ngs[,]⏎␣␣␣`<br>`fdd3,␣0xfdd4,␣0xfdd5,␣0x[f]dd6,` |
| 1318 | 0.50 % | d (65 %) | , (20 %) | `0x00c1␣->␣BOX␣DRAWINGS␣L[I]GHT␣`<br>`␣␣␣0x1a,␣0x1b,␣0x1c,␣0x1[d],␣0x`<br>`005[d]␣->␣` |
| 1371 | 4.30 % | ␣ (85 %) | ␣ (65 %) | `⏎␣␣␣␣def␣__init__(self,␣[*],␣ob`<br>`nal_kwargs(self,␣dest,␣*[*]kwar`<br>`onal_kwargs(self,␣dest,␣[*]*kwa` |
| 1484 | 0.65 % | O (55 %) | U (50 %) | `␣SINGLE␣AND␣HORIZONTAL␣D[O]UBLE`<br>`AGE␣\|␣_winapi.PIPE_READM[O]DE_M`<br>`4,␣␣␣␣␣#␣␣BOX␣DRAWINGS␣D[O]UBLE` |
| 1496 | 1.93 % | , (90 %) | ␣ (90 %) | `er):⏎␣␣␣␣def␣encode(self[,]␣inp`<br>`handle_cancelled(self[,]␣han`<br>`␣=␣pgo⏎⏎␣␣␣␣def␣log(self[,]␣lin` |
| 1568 | 5.86 % | : (80 %) | ⏎ (85 %) | `g,␣args)[:]⏎␣␣␣`<br>`dec,codecs.StreamWriter)[:]⏎␣␣␣`<br>`coder)[:]⏎␣␣␣` |
| 1649 | 1.67 % | ␣ (60 %) | R (40 %) | `RAWINGS␣DOUBLE␣V[E]RTIC`<br>`#␣␣BOX␣DRAWINGS␣DOUBLE␣V[E]RTIC`<br>`DRAWINGS␣LIGHT␣UP␣AND[␣]RIGH` |
| 1664 | 0.98 % | d (80 %) | e (75 %) | `rror⏎⏎␣␣␣␣#␣[D]ebug`<br>`ion⏎⏎␣␣␣␣[d]ef␣i`<br>`␣"args",␣"keywords",␣"__[d]ict_` |
| 1717 | 1.28 % | e (100 %) | ␣ (80 %) | `se:⏎␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣rais[e]␣Run`<br>`␣␣␣␣␣␣␣␣"""⏎␣␣␣␣␣␣␣␣rais[e]␣Not`<br>`FATAL):⏎␣␣␣␣␣␣␣␣␣␣␣␣rais[e]␣Val` |
| 1754 | 1.93 % | x (100 %) | 0 (50 %) | `IGN⏎␣␣␣␣0[x]00b1`<br>`␣HALF␣BLOCK⏎␣␣␣␣0[x]00df`<br>`␣␣0[x]00c9` |
| 1854 | 0.53 % | e (80 %) | t (65 %) | `_class__):⏎␣␣␣␣␣␣␣␣␣␣␣␣r[e]turn`<br>`esult)⏎␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣r[e]sult`<br>`⏎␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣self.wa[r]ning` |
| 1906 | 0.24 % | o (100 %) | s (80 %) | `␣␣#␣namespace⏎⏎␣␣␣␣␣␣␣␣p[o]siti`<br>`e␣=␣self._handle_queue.p[o]p(0)`<br>`␣␣␣(lnum,␣pos),␣(lnum,␣p[o]s+1)` |
| 1974 | 1.51 % | ⏎ (60 %) | ␣ (55 %) | `er):⏎␣␣␣␣codec␣=␣codec⏎⏎[c]lass`<br>`ors,decoding_table)[0]⏎⏎[c]lass`<br>`errors,decoding_table)⏎⏎[c]lass` |
| 2034 | 7.90 % | ␣ (50 %) | ␣ (70 %) | `␣␣␣␣␣␣␣␣self.close␣=␣uti[l].Fin`<br>`␣␣␣␣␣␣self._unlink␣=␣uti[l].Fin`<br>`␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣'␣wit[h]␣` |
