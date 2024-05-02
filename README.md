
### 데이터 셋
* [input dataset](https://t1h0q-my.sharepoint.com/personal/csbhr_t1h0q_onmicrosoft_com/_layouts/15/onedrive.aspx?ga=1&id=%2Fpersonal%2Fcsbhr%5Ft1h0q%5Fonmicrosoft%5Fcom%2FDocuments%2FRelease%2FFFHQ%2DUV%2Fdataset%2Fffhq%2Duv%2Fuv%2Dmaps)
                
* [target dataset](https://github.com/ubisoft/ubisoft-laforge-FFHQ-UV-Intrinsics)

### 
페어 맞추기 위해 10000건만 사용


### 구조 
입력 -> Resnet50 (as Encoder) -> MyDecoder -> 출력

### 데이터 shape
 * input = 3, w, h
 * target = 4, 3, w, h (디퓨즈, 라이트 노말라이제이션, 노말의 RGB 파일과 AO, SPEC, 투명도 그레이 스케일을 rgb 각 채널에 분배하여 총 4장의 이미지로 만듬)

|제목|내용|설명|
|---------------------|-|---|
|diffuse              |0|RGB|
|light_normalized     |1|RGB|
|normal               |2|RGB|
|Specular             |3|R  |
|Ambient Occlusion    |3| G |
|Translucency         |3|  B|
