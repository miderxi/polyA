> 1. PolyAdenylation mechanism

![图片](https://user-images.githubusercontent.com/41559035/175261562-794a7430-2b9b-43ed-b7fb-393ceace90a6.png)
![图片](https://user-images.githubusercontent.com/41559035/175261583-e10ad8c7-41ab-4944-85fa-a6e6fb64a960.png)
[1]Salamov, Asaf A., and Victor V. Solovyev. "Recognition of 3'-processing sites of human mRNA precursors." *Bioinformatics* 13.1 (1997): 23-28.

[1]Salamov, Asaf A., and Victor V. Solovyev. "Recognition of 3'-processing sites of human mRNA precursors." *Bioinformatics* 13.1 (1997): 23-28.


>2. twelve class polyA predict score（10 fold validation）


![图片](https://user-images.githubusercontent.com/41559035/175261716-592bf524-d6dc-4ef8-bed9-4c99b8933938.png)
![图片](https://user-images.githubusercontent.com/41559035/175261729-f71cbc7a-c72f-4bf9-8e99-a07263ecdc5a.png)
![图片](https://user-images.githubusercontent.com/41559035/175261744-38ea944f-ba5e-4c3e-a95d-4811c5f06d6b.png)

>3. comparation of AATAAA polyA site

| AATAAA    | Sp（%）     | Sn（%）     | Acc（%）    |
| ----------- | ------------- | ------------- | ------------- |
| Our CNN   | 84.69±3.55 | 85.57±2.45 | 84.97±1.77 |
| ANN       | 80.55       | 80.55       | 82.06       |
| SVM       | 37.2-71.0   | 74.6-96.7%  | 61.36       |
| polyapred | 75.8-95.7   | 56.0-93.3   | 67.5-93.3   |

>4. Grad-CAM visual
> We confirm the mechanism that the human polyA signal is  determinded by 15 bp sequece of its upsteam and downstream  by conputation.


![图片](https://user-images.githubusercontent.com/41559035/175262018-2521d202-1985-4ca8-ae56-492faff1f32a.png)


