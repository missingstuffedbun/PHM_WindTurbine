"""模型服务包：把实验结果中表现最好的模型包装成对外推理服务。

三层结构，可独立使用：

    serve/registry.py   从实验结果目录挑选最优 run（指标排序 + 权重定位）
    serve/preprocess.py 推理侧输入构造（原始工程量纲 -> 模型输入张量）
    serve/predictor.py  加载权重并提供 predict()（纯 torch，无 Web 依赖）
    serve/server.py     FastAPI 封装，暴露 HTTP 接口（需要 fastapi / uvicorn）

不装 Web 依赖时，前三层仍可作为库直接调用：

    from serve.predictor import BestModelPredictor
    predictor = BestModelPredictor.from_results("results")
    predictor.predict([{...}, ...])   # 一个窗口的原始采样
"""
