# Brain-tumor-segmentation
Segmenting brain tumor by using various networks from these papers:
1. [U-net: Convolutional networks for biomedical image segmentation](https://link.springer.com/content/pdf/10.1007/978-3-319-24574-4_28.pdf)
2. [ResUNet-a: A deep learning framework for semantic segmentation of remotely sensed data](https://pdf.sciencedirectassets.com/271826/1-s2.0-S0924271620X00037/1-s2.0-S0924271620300149/main.pdf?X-Amz-Security-Token=IQoJb3JpZ2luX2VjEO7%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLWVhc3QtMSJHMEUCIEjF9FPeSack9uLwI0WlVkGIaqnnuCpzMYAFnva5xWWXAiEA%2Bt%2BWyhxMTf1XEsYHNB%2B726x8grkhYmWPXwnRlaLZPf4qgwQI5%2F%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FARAEGgwwNTkwMDM1NDY4NjUiDF32h0dWMDZghdtr%2BSrXAxJ35ba1%2F9Uw1xBN%2FS%2Btl7RKL3F8lBWYPUaELbhXbVrD%2B41DNjK2CZn8f%2BuC%2FfPJ5H79qQIQbiTXYgXAgWPR2UjQeeYjfkR8SuBJp5jPWw%2B0g%2FvzAMTbma9apzZl20Ih2npZ59sA06nXSFzfDTmws4GIshADxx3uC%2BToDNldF66pUhuzc5I6ZSp81IcGX%2FgkjmrGPYr0%2F6kahJGPOUueAt%2B2XFGfAEo4BOZJGQzLgmplDEIeyXOi4ralAFQFmH9LaMYQW2cMV%2F8oQu221inoiQr6WodLRA8CF2iMbMjsrOiF3h8J7yMJCDAkxTU4idSU5ExOabQHouKVRWqZwVIDHjU6fGnt3TApJqIRCK4c%2FaAqel489nwWvRyRF%2FXvrBI2AXZ3A6gVEIW3yXJIIqvg1I3BmD4o3CX7SewMw%2BIbZqBkQW0yxPltCi%2FE9hSKLt3Ksat24nMAyVUKeKDpMzFGqx5qXF0x70zIDNfO6BQC3nKq7crHaAgpGB5yIUMw8HcTd6ty8Hbs%2FIEcEpXfNTVJ2GK1KPKSS88k0CROGCHZ3MJtH5P2v1NVi5A%2BSak87wlbFKqYDtVk6083fb5H%2FOP6LUx74pbrzr3PuzGsJ%2BMT0MAcgVRSWeXDuDDqkLWOBjqlATVJeOgO5X87wsILenZD%2F4B5ga7ksjO7DzKSmVmoAvFR6epz96TAoawXmJGzWSUtJHlqQtDo2xD0D6a1XJq9m30hPAy1G8xj1aRN4nw59fGZXnLgGFbPu0WMZD6hopazUZ37asi5gx5YA14OFXd688fnoKyka71z4bcqZte0HRrtHqpTVLSy91xa%2BAeBLERadoEi5Ukf09bpXhejG3PWDjCgC41arA%3D%3D&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20211230T071801Z&X-Amz-SignedHeaders=host&X-Amz-Expires=300&X-Amz-Credential=ASIAQ3PHCVTYQEC6SUEY%2F20211230%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=bf16da874b01546060a168edb1b79bba3fa9d89b79a9d15ec32239741fd36aa9&hash=6a8cddd43ed2237ef7120c8d2c8e2d99845e5f5206fd874467e321b2cceefa51&host=68042c943591013ac2b2430a89b270f6af2c76d8dfd086a07176afe7c76c2c61&pii=S0924271620300149&tid=spdf-9f68aa6a-a703-4431-9f25-c64a80388e16&sid=c953467d77871143cf79c8d31267574a5548gxrqa&type=client)
3. [Resunet++: An advanced architecture for medical image segmentation](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8959021&casa_token=0Dwkq5BXH3YAAAAA:fZQOmiFgwRrWc0QQYq33U-QGSzQKC0S-Fl_k6ljId0diro56fx723uggF7RemNj8GE9-JFuG1ek&tag=1)

## Project Topic & Goal
The main goal of this project is the brain tumor segmentation.
- As the use of deep learning in the medical field is gradually increasing, high diagnostic performance is being derived.
- Furthermore, in order to utilize medical data, the need to automate data analysis and purification, which was previously handled by experts, is gradually increasing.
- If a high-accuracy and generalizable model can be used, the time that radiologists spend on data analysis can be effectively reduced.

## Dataset
We downloaded the [dataset from Kaggle](https://www.kaggle.com/mateuszbuda/lgg-mri-segmentation).

1. Data format: .tif
2. Total: 110 dataset (Train/Validation-100, Test-10)
3. Ground-truth data: Mask data

## Method & Performance
In order to select a model that can improve performance, we compared three models with promising performance in the segmentation task.
![brain_net](https://user-images.githubusercontent.com/57162425/147729390-4d912307-d379-4280-9f8a-037aafa422df.png)

### Qualitative evaluation (segmented mask visualization)
![brain_res](https://user-images.githubusercontent.com/57162425/147729386-ee0790cf-ff3c-4398-9a0f-cd9840690ced.png)

### Quantitative evaluation
![brain_1](https://user-images.githubusercontent.com/57162425/147729392-b6bb661c-e784-4c21-be1e-c183a4ebe197.png)

