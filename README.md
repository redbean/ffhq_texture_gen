## 1 FFHQ Image 2 PBR Texture Sets Generator


### 데이터 셋
* [input dataset : ffhq](https://t1h0q-my.sharepoint.com/personal/csbhr_t1h0q_onmicrosoft_com/_layouts/15/onedrive.aspx?ga=1&id=%2Fpersonal%2Fcsbhr%5Ft1h0q%5Fonmicrosoft%5Fcom%2FDocuments%2FRelease%2FFFHQ%2DUV%2Fdataset%2Fffhq%2Duv%2Fuv%2Dmaps)
                
* [target dataset : Ubisoft-laforge-FFHQ-UV-Intrinsics](https://github.com/ubisoft/ubisoft-laforge-FFHQ-UV-Intrinsics)

### 
데이터 셋 사이즈 : 페어 맞추기 위해 10000건만 사용

### 구조 
입력 [3,w,h] -> Encoder -> RVQ -> Decoder -> 출력 [12,w,h]

### 데이터 shape
 * input = 3, w, h
 * target = 12, w, h 디퓨즈, LN, Normal, AO, SPEC, TRANSLUCENCY, 의 모든 채널을 1개의 쉐입으로 정리

    
|제목|InChan-순서|설명|
|---------------------|---|---|
|diffuse              |0-2|RGB|
|light_normalized     |3-5|RGB|
|normal               |6-8|RGB|
|Specular             |9  |R  |
|Ambient Occlusion    |10 | G |
|Translucency         |11 |  B|



----old----
### 구조 
입력 -> Resnet50 (as Encoder) -> MyDecoder -> 출력

* 0605 update -->old(디퓨즈, 라이트 노말라이제이션, 노말의 RGB 파일과 AO, SPEC, 투명도 그레이 스케일을 rgb 각 채널에 분배하여 총 4장의 이미지로 만듬)

|제목|내용|설명|
|---------------------|-|---|
|diffuse              |0|RGB|
|light_normalized     |1|RGB|
|normal               |2|RGB|
|Specular             |3|R  |
|Ambient Occlusion    |3| G |
|Translucency         |3|  B|

