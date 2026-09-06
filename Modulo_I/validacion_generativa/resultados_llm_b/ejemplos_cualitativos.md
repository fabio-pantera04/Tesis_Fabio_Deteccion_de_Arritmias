# Ejemplos cualitativos del Modulo LLM-B

Muestra de generaciones vs referencia PTB-XL, 3 casos por clase para cada K.


## K = 0


### NORMAL (K=0)

**Ejemplo 1** (ecg_id 6119):

- **Referencia PTB-XL**: sinus rhythm. normal ecg.

- **Generada por LLM-B**: normal ecg.

**Ejemplo 2** (ecg_id 10823):

- **Referencia PTB-XL**: sinus rhythm normal ecg.

- **Generada por LLM-B**: normal ecg.

**Ejemplo 3** (ecg_id 11014):

- **Referencia PTB-XL**: sinus rhythm. normal ecg.

- **Generada por LLM-B**: normal ecg


### PAC (K=0)

**Ejemplo 1** (ecg_id 6557):

- **Referencia PTB-XL**: premature atrial contraction(s). sinus rhythm. non-specific intraventricular delay. q waves in v2 suggesting anteroseptal infarct,    probably old. t waves are low or flat in i, v5,6 and inverted in avl. non-specific but consistent with myocardial ischaem

- **Generada por LLM-B**: premature atrial complexes

**Ejemplo 2** (ecg_id 12676):

- **Referencia PTB-XL**: premature atrial contraction(s). sinus rhythm. compared with tracing of 1:8:92. there is now slight st segment depression in i, avl, v4,5,6. myocardial ischaemia cannot be excluded.

- **Generada por LLM-B**: premature atrial contraction. typical morphology.

**Ejemplo 3** (ecg_id 12717):

- **Referencia PTB-XL**: premature atrial contraction(s). sinus rhythm. left axis deviation. left anterior fascicular block. voltages are high in chest leads suggesting lvh.

- **Generada por LLM-B**: premature atrial contractions


### NSR (K=0)

**Ejemplo 1** (ecg_id 12049):

- **Referencia PTB-XL**: sinus rhythm. possible old inferior infarct. non-specific t wave flattening in v2,3,4.

- **Generada por LLM-B**: normal sinus rhythm.

**Ejemplo 2** (ecg_id 12075):

- **Referencia PTB-XL**: sinus rhythm. normal ecg.

- **Generada por LLM-B**: normal sinus rhythm. normal p wave and qrs complex morphology.

**Ejemplo 3** (ecg_id 12012):

- **Referencia PTB-XL**: sinus rhythm. left ventricular hypertrophy. otherwise no definite pathology.

- **Generada por LLM-B**: normal sinus rhythm, typical morphology.


### AFIB (K=0)

**Ejemplo 1** (ecg_id 17657):

- **Referencia PTB-XL**: atrial fibrillation. voltages are high in chest leads suggesting lvh. st segments are depressed in i, ii, v5,6. this may be due to lv strain or ischaemia.

- **Generada por LLM-B**: atrial fibrillation; irregularly irregular rhythm; no discernible p waves; fibrillatory waves

**Ejemplo 2** (ecg_id 16475):

- **Referencia PTB-XL**: atrial fibrillation. otherwise no definite pathology.

- **Generada por LLM-B**: atrial fibrillation. irregularly irregular rhythm with absence of discrete p waves.

**Ejemplo 3** (ecg_id 17851):

- **Referencia PTB-XL**: atrial fibrillation. premature ventricular contraction(s). low limb lead voltage. non-specific t wave flattening in limb leads and    v4,5,6.

- **Generada por LLM-B**: atrial fibrillation; irregularly irregular rhythm.


## K = 10


### NORMAL (K=10)

**Ejemplo 1** (ecg_id 10920):

- **Referencia PTB-XL**: sinus rhythm. normal ecg.

- **Generada por LLM-B**: sinus rhythm. normal ecg.

**Ejemplo 2** (ecg_id 6729):

- **Referencia PTB-XL**: sinus rhythm. normal ecg.

- **Generada por LLM-B**: sinus rhythm. normal ecg.

**Ejemplo 3** (ecg_id 1241):

- **Referencia PTB-XL**: sinus rhythm. normal ecg.

- **Generada por LLM-B**: sinus rhythm. normal ecg.


### PAC (K=10)

**Ejemplo 1** (ecg_id 5103):

- **Referencia PTB-XL**: premature atrial contraction(s), trigeminy. sinus rhythm. r-s transition zone in v leads displaced to the right. left ventricular hypertrophy. st segments are depressed in i, avl, v5,6. t waves are low or flat in limb leads and v5,6. this may be due to lv

- **Generada por LLM-B**: premature atrial contraction(s). sinus rhythm. otherwise normal ecg.

**Ejemplo 2** (ecg_id 964):

- **Referencia PTB-XL**: premature atrial contraction(s). sinus rhythm. left bundle branch block, this is most commonly due to    ischaemic heart disease.

- **Generada por LLM-B**: premature atrial contraction(s). sinus rhythm. minor non-specific st-t wave changes.

**Ejemplo 3** (ecg_id 12593):

- **Referencia PTB-XL**: premature atrial contraction(s). sinus rhythm. posssible left atrial enlargement. left axis deviation. left anterior fascicular block. tiny r waves in v2,3 suggesting old anteroseptal  infarct. st segments are depressed in i, avl. t waves are flat in i an

- **Generada por LLM-B**: premature atrial contraction(s). sinus rhythm. otherwise normal ecg.


### NSR (K=10)

**Ejemplo 1** (ecg_id 12053):

- **Referencia PTB-XL**: sinus rhythm. possible old inferior infarct.

- **Generada por LLM-B**: sinus rhythm. normal ecg.

**Ejemplo 2** (ecg_id 12048):

- **Referencia PTB-XL**: sinus rhythm. non-specific intraventricular delay. otherwise normal ecg.

- **Generada por LLM-B**: sinus rhythm. normal ecg.

**Ejemplo 3** (ecg_id 1322):

- **Referencia PTB-XL**: sinus rhythm. voltages are high in limb leads suggesting lvh. possible left atrial enlargement. otherwise no definite pathology.

- **Generada por LLM-B**: sinus rhythm. normal ecg.


### AFIB (K=10)

**Ejemplo 1** (ecg_id 2964):

- **Referencia PTB-XL**: atrial fibrillation. left axis deviation. non-specific intraventricular block.

- **Generada por LLM-B**: atrial fibrillation. st segments are depressed in ii, iii, avf, v4,5,6. non-specific but consistent with myocardial ischaemia and digitalis effect.

**Ejemplo 2** (ecg_id 12297):

- **Referencia PTB-XL**: atrial fibrillation. incomplete right bundle branch block. widespread st segment depression and t wave inversion. findings are likely to be due to ischaemic heart    disease. the age of the changes is uncertain.

- **Generada por LLM-B**: atrial fibrillation with variable ventricular response. otherwise no definite pathology.

**Ejemplo 3** (ecg_id 1799):

- **Referencia PTB-XL**: atrial fibrillation. however, rhythm is basically regular, indicating that it derives from a supraventricular focus. some beats show non-specific intraventricular block. left axis deviation. left anterior fascicular block. t waves are flat in limb leads a

- **Generada por LLM-B**: atrial fibrillation. irregular ventricular response.


## K = 20


### NORMAL (K=20)

**Ejemplo 1** (ecg_id 10718):

- **Referencia PTB-XL**: sinus rhythm. normal ecg.

- **Generada por LLM-B**: sinus rhythm. normal ecg.

**Ejemplo 2** (ecg_id 16655):

- **Referencia PTB-XL**: sinus rhythm normal ecg

- **Generada por LLM-B**: sinus rhythm. normal ecg.

**Ejemplo 3** (ecg_id 14133):

- **Referencia PTB-XL**: sinus rhythm. normal ecg.

- **Generada por LLM-B**: sinus rhythm. normal ecg.


### PAC (K=20)

**Ejemplo 1** (ecg_id 14354):

- **Referencia PTB-XL**: premature atrial contraction(s). sinus rhythm. left atrial enlargement. possible old inferior infarct.

- **Generada por LLM-B**: premature atrial contraction(s). sinus rhythm.

**Ejemplo 2** (ecg_id 10378):

- **Referencia PTB-XL**: premature atrial contraction(s). sinus rhythm. otherwise normal ecg. Edit: NORM 80, (NORM 100)

- **Generada por LLM-B**: premature atrial contraction(s). sinus rhythm. otherwise normal ecg.

**Ejemplo 3** (ecg_id 2853):

- **Referencia PTB-XL**: premature atrial contraction(s). sinus rhythm. otherwise normal ecg.

- **Generada por LLM-B**: premature atrial contraction(s). sinus rhythm. otherwise normal ecg.


### NSR (K=20)

**Ejemplo 1** (ecg_id 12049):

- **Referencia PTB-XL**: sinus rhythm. possible old inferior infarct. non-specific t wave flattening in v2,3,4.

- **Generada por LLM-B**: sinus rhythm. normal ecg.

**Ejemplo 2** (ecg_id 1070):

- **Referencia PTB-XL**: sinus rhythm. premature ventricular contraction(s). voltages are high in limb leads suggesting lvh. st segments are depressed in i, avl. t waves are flat in limb leads and inverted in v5,6. findings are likely to be due to ischaemic heart    disease.

- **Generada por LLM-B**: sinus rhythm. normal ecg.

**Ejemplo 3** (ecg_id 2976):

- **Referencia PTB-XL**: sinus rhythm. prolonged pr interval. otherwise no definite pathology.

- **Generada por LLM-B**: sinus rhythm. normal ecg.


### AFIB (K=20)

**Ejemplo 1** (ecg_id 14088):

- **Referencia PTB-XL**: atrial fibrillation with rapid ventricular response. st segments are depressed and t waves inverted in ii,    iii, avf, v3-6. this may be due to lv strain or ischaemia. the age of the changes is uncertain.

- **Generada por LLM-B**: atrial fibrillation.

**Ejemplo 2** (ecg_id 307):

- **Referencia PTB-XL**: atrial fibrillation. qs complexes in v2 and tiny r waves in v3,4. st segments are depressed in i, ii, avl, v5,6. t waves are inverted in i, avl, v5,6. consistent with ischaemic heart disease with old anteroseptal myocardial infarction. it is difficult to

- **Generada por LLM-B**: atrial fibrillation.

**Ejemplo 3** (ecg_id 14316):

- **Referencia PTB-XL**: premature ventricular contraction(s). atrial fibrillation. rightward axis. right bundle branch block. possible old inferior myocardial infarct.

- **Generada por LLM-B**: atrial fibrillation. non-specific st-t wave changes in i, ii, iii, avf, v6.
