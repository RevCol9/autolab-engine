# 可选半自动标注引擎

训练平台不会随应用包分发 SAM3、Locate Anything 或自有模型的权重。三种引擎是彼此独立的后端适配器；前端只调用训练后端，绝不保存或展示服务地址和令牌。

## 配置

复制 `annotation-engines.env.example` 中需要的变量到训练后端运行环境。`application-prod.yml` 默认启用 104 上的 Autolab 自有模型服务，SAM3 和 Locate Anything 默认关闭；所有值均可由环境变量覆盖。当前 104 的启动脚本实际运行默认 `dev` profile，因此部署时由 `/project/niii/boot-vision/backend/auto_start_boot-vision.sh` 显式导出 `SELF_MODEL_ENABLED=true` 和 `SELF_MODEL_BASE_URL=http://10.65.48.104:21010`，不能只依赖 `application-prod.yml`。

104 的后端制品更新可使用根仓库 `deploy/scripts/deploy-autolab-annotation-104.sh`。脚本要求先把启动 JAR 和业务 JAR 分别上传为 `boot-vision.jar.autolab.new`、`lib/niii-boot-biz-2.4.2.jar.autolab.new`，并传入两个 SHA-256；它会备份现有制品和启动脚本，校验后替换，健康检查失败时自动回滚。

部署后使用根仓库 `deploy/scripts/verify-autolab-annotation-104.sh` 做只读验收。通过环境变量提供管理员和数据库密码；脚本检查训练后端/Autolab 健康状态、四个算法的引擎状态，并选择一张服务器上可读的正样本请求建议，不调用保存接口。

- 未配置或 `*_ENABLED=false`：接口返回 `not_configured`。
- 已配置但未提供可用推理服务：接口返回 `not_deployed` 或 `unavailable`。
- 请求超时：接口返回 `timeout`，现有建议和手工标注不会丢失。

SAM3 和 Locate Anything 的 API 契约由 autolab-engine 推理服务（`:21010`）提供，见 `docs/06-locate-sam3.md`、`docs/08-segment-api.md`。

## Autolab 自有模型

训练后端通过 `SELF_MODEL_BASE_URL` 访问 Autolab，前端不会直接连接推理服务。每次建议请求依次执行：

1. `POST /api/models/{model_key}/load`
2. multipart `POST /api/predict`，字段为 `image`、`model_key` 和 `box_format=cxcywh_pct`

生产映射在训练后端 `application-prod.yml` 的 `niii.annotation-engines.self-model.mappings` 中维护：

- `Smoke` -> `smoke_flame_fire_extinguisher_person_helmet`，只接收远端类别 `0`。
- `Flame` -> 同一组合模型，只接收远端类别 `1`，并改为平台类别 `0`。
- `Helmet` -> `workwear_helmet`，类别 `0/1` 原样映射。
- `Fall_Detection` -> `fall_nofall_bending`，类别 `0/1/2` 原样映射。

服务返回的 `cx/cy/w/h` 是 `0..100` 百分比，训练后端除以 `100` 后生成归一化 `rect`。没有映射的类别被过滤；合法空检测仍视为成功。平台只上传数据库样本指向且位于 `jeecg.path.upload` 内的文件，日志和接口错误不得包含令牌或完整服务器路径。

Autolab 按固定 `model_key` 加载部署模型，目前不接收平台 PT 文件或版本 ID。版本选择保留为平台侧追踪信息；没有本地 PT 版本时不阻断已映射算法调用。

## 运行验收

1. 不部署任何引擎时，`GET /niii/algorithm/annotation/engines` 显示非阻塞状态，矩形、多边形、魔法棒和保存仍可使用。
2. 对 `Smoke`、`Flame`、`Helmet`、`Fall_Detection` 调用自有模型，确认先加载对应模型、再返回仅包含已映射类别的建议；建议在用户接受前仅存在浏览器内。
3. Locate Anything 的矩形建议仅在 SAM3 可用时可细化为分割多边形。
4. 训练预检和启动均返回 `positiveLabeled`、`negativeAvailable`、`negativeLimit`、`negativeAdopted`，并保证 `negativeAdopted <= floor(positiveLabeled / 3)`。
