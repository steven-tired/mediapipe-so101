# PV 工具可读性重构计划

范围:`integrations/pressurevision/tools/record_so101_pv_ee.py`(1887 行)与
`integrations/pressurevision/tools/deploy_so101_grip_ee.py`(2033 行)。
目标是让新人能读懂,不是让它更"现代"。

**硬约束**

- 这条控制路径已过硬件验证。批次 1–4 的每一处改动都必须是"看一眼就知道没改行为"。
- 不重写控制逻辑;不为风格统一去动 `ee_controller.py` / `ee_control.py`。
- 批次 5b 会改变运行时行为,必须配一次物理闸门,结果记进 `docs/RELEASE_AUDIT.md`。

**执行顺序**:1 → 2 → 3 → 4 → 5a → 5b。前四批可以合成一个 PR;5b 单独一个 PR,
因为它需要闸门记录。

---

## 批次总览

| # | 内容 | 风险 | 闸门 |
|---|---|---|---|
| 1 | 纯改名 | 零 | 否 |
| 2 | 删死代码 | 零 | 否 |
| 3 | 纯搬运抽函数 | 零(剪贴) | 否 |
| 4 | 注释与分节 | 零(无代码) | 否 |
| 5a | 文案与入参校验修正 | 低 | 否 |
| 5b | 退出路径与按键语义修复 | 高 | **是** |

---

## 批次 1 — 纯改名

### 1.1 `record_so101_pv_ee.py:1588` — `preview` 同时是对象和布尔

**问题**:同一个调用里两个 `preview` 指不同的东西。

```python
preview = PressureVisionPreviewSource(args.pv_preview_share)      # 1588
wait_for_continuous_hand_tracking(source, robot, preview,          # 1593 → 形参 pv_preview
                                  preview=not args.no_preview)     # 1594 → 形参 preview(bool)
PVRecorderTeleop(cfg, controller, pv, source, robot, preview,      # 1652 → 形参 pv_preview
                 sidecar, evidence, preview=not args.no_preview)   # 1655
```

被调方(`895`、`1058`)内部分得很清楚:`pv_preview` 是 PV 画面源,`preview` 是要不要开
cv2 窗口。只有调用方这一侧把两者都叫 `preview`,且写在相邻两行。

**改法**:`1588` 改名为 `pv_preview`,连带更新 `1589`、`1593`、`1652` 三处引用。
共 4 处标识符替换,不动任何形参名。

### 1.2 `record_so101_pv_ee.py:1242` — `anchor` 在同一函数里被读两次

**问题**:`1216` 的 `anchor` 用于 evidence 事件判定,并在 `1219` 起被消费;`1242` 在
preview 分支里**重新读一次** `self.pv.adjustment_anchor_target` 覆盖它。同名同义但是
不同时刻的采样,读者会以为是同一个值。

**改法**:`1242` 改名为 `banner_anchor`,连带 `1244`、`1246` 的两处格式化引用。
**不要**删掉这次重读去复用 `1216` 的值——两次读之间隔着 `_pending_sample` 装配和锁状态
判定,那样会改变行为。

### 1.3 `record_so101_pv_ee.py:1554` — `dataset_mode` 三行内被就地覆盖

**问题**:`"reset_empty"` 读进来,reset 完覆盖成 `"create"`,而 `checked["dataset_mode"]`
里仍是旧值;`1690` 的 `if dataset_mode == "resume"` 读的是覆盖后的。一个名字两个含义。

**改法**:

```python
dataset_mode = checked["dataset_mode"]
if dataset_mode == "reset_empty":
    reset_empty_dataset_root(dataset_root)
open_mode = "create" if dataset_mode == "reset_empty" else dataset_mode
```

`1690` 改读 `open_mode`。两个名字各自恒定。

### 1.4 `record_so101_pv_ee.py:1693` — `dataset` 与 `recording_dataset`

**问题**:`dataset` 是 `LeRobotDataset`,`recording_dataset` 是包着它的
`PVTeachingDatasetView`,而 `record_loop` 的形参也叫 `dataset=`,于是 `1719` 读作
`dataset=recording_dataset`。两个名字在同一作用域都自称 dataset:`save_episode` /
`clear_episode_buffer` 走前者,`add_frame` 走后者。

**改法**:`recording_dataset` → `teaching_view`(`1693`、`1719` 两处)。

---

## 批次 2 — 删死代码

### 2.1 `record_so101_pv_ee.py:1550` 与 `1827` — 局部 `status` 从未被读取

**问题**:`status = "aborted"`(1550)与
`status = "normal" if not events["stop_recording"] else "stopped"`(1827)两次赋值,
全函数没有任何读取点——`1829` 的 `evidence.close(status="kept" if keep else "rolled_back")`
用的是另一套取值。代码看起来 evidence 里有 normal/stopped/aborted 这个维度,实际没有。

**改法**:删掉 `1550` 和 `1827` 两行赋值,在 `1827` 位置留一行注释:

```python
# stop 与否体现在每次尝试的 outcome 记录里,不再单独汇总到 manifest。
```

**不要**顺手把它接进 `evidence.close()`——那会改 evidence 的输出 schema,属于另一件事。

### 2.2 `record_so101_pv_ee.py:1696` — `resources.callback(lambda: None)` 是空注册

**问题**:夹在 `dataset.finalize` 与 `teleop.connect()` 之间的空回调。读者必然停下来想
"这里本来该注册什么"。

**改法**:删除该行。若它其实是刻意的顺序占位(保证之后注册的回调排在 `dataset.finalize`
之后触发),则保留并补一行注释说明该意图——提交前先用 `git log -L1696,1696:` 确认它的
来历,不要凭猜。

---

## 批次 3 — 纯搬运抽函数

全部是整段剪贴到模块级函数,不改一行内部代码。

### 3.1 `deploy_so101_grip_ee.py:1104-1252` → `build_parser()`

**问题**:`main()` 共 931 行,开头 149 行是连续的 `ap.add_argument`。读者要滚过全部才能
看到第一行真实逻辑。

**改法**:整段剪到模块级 `def build_parser() -> argparse.ArgumentParser:`,末尾 `return ap`。
`main()` 里:

```python
ap = build_parser()
args = ap.parse_args()
```

### 3.2 `deploy_so101_grip_ee.py:1254-1347` → `validate_args(ap, args)`

**问题**:94 行、38 条 `ap.error`,`--paired-boundaries` / stall ramp / grip intervention
三组互斥规则彼此穿插,没有分组。

**改法**:剪成 `def validate_args(ap: argparse.ArgumentParser, args: argparse.Namespace) -> None:`。
`ap` 作参数传入,`ap.error` 语义不变(仍是 `SystemExit(2)`)。
注意 `1342` 的 `needs_feedback` 在后面 `1394` 还要用——把它作为返回值:
`needs_feedback = validate_args(ap, args)`,或在 `main` 里就地重算一次(表达式是纯的)。
**推荐前者**,避免同一条件写两遍。

### 3.3 `deploy_so101_grip_ee.py:1516-1602` → `build_controllers(args, scorer)`

**问题**:五个可选控制器(correction / paired / stall / intervention / candidate)各自
15–25 行构造 + `print` 横幅,夹在 `robot.connect()` 和主循环之间,把"连上机械臂之后做什么"
这条主线切断了。

**改法**:剪成一个函数,返回这五组对象。返回形式用一个模块级
`@dataclass` 承载(`correction_recorder / correction_source / correction_toggle / paired /
paired_operator / stall_tighten / stall_operator / grip_intervention / grip_candidate_trial`),
比返回 9 元组安全。`main` 里解包到原来的同名局部变量,循环体一行不动。

### 3.4 `deploy_so101_grip_ee.py:1920-2029` → `_shutdown(...)`

**问题**:`finally` 块 110 行:5 个 `close()` + evidence 关闭 + 两段结果打印 +
relax/disconnect。

**改法**:整段剪出。所有引用的名字在 `finally` 里都已可见,按值传参即可。
`finally:` 体内只剩一行调用。

> 抽完这四刀,`main()` 剩约 290 行:connect → evidence → ramp → while → 异常分支,
> 骨架一屏可见。

### 3.5 `record_so101_pv_ee.py:1728 / 1756 / 1770 / 1787` → `_discard_attempt(...)`

**问题**:同一个"丢弃这次尝试"的四联动重复了四次:

```python
evidence.outcome(system_outcome_record(attempt=number, status=…, reason=…,
                                       review_video=…, review_timeline=…,
                                       evidence_root=evidence.path))
dataset.clear_episode_buffer()
teleop.end_episode(number, …)
```

四份里 `status` 与 `end_episode` 标签**不成对**——`"invalid"`/`"discarded_pv_fault"`、
`"aborted"`/`"aborted"`、`"aborted_review"`/`"aborted_review"`、`"rerecord"`/`"rerecord"`。
新人无法判断这个不一致是有意的还是抄漏的。

**改法**:

```python
def _discard_attempt(evidence, dataset, teleop, *, number, status, reason,
                     end_label, review_video, review_timeline) -> None:
```

四处各变一行,标签差异变成四个显式实参,一眼可比。**不要**在这一步顺手把标签改成一致——
那是行为改动,先问清楚 `"invalid"` → `"discarded_pv_fault"` 是不是有意的。

### 3.6 `record_so101_pv_ee.py:1531-1546` → `_open_evidence_session(args, checked)`

**问题**:evidence 会话的创建、拒绝覆盖非空目录、哈希装配,共 16 行,在 `try:` 之前,
与数据集无关。

**改法**:剪成 `def _open_evidence_session(args, checked) -> EvidenceSession:`。纯输入到输出。

---

## 批次 4 — 注释与分节(不改代码)

### 4.1 `deploy_so101_grip_ee.py:1712` — 一个动作向量的五个名字

**问题**:循环里同时活着 `a`(工作值,被就地改写四次)、`action`(policy 张量,1702)、
`predicted_action`(1712 的快照)、`planned_action`(1806 的 dict)、`bus_action`
(1801 的回读)。`a` 的语义随行数漂移:1673 是 policy 输出 → 1713 加了 close offset →
1735-1786 夹爪被某个控制器改写 → 1804 身体分量可能被 `hold_body_action` 换掉。

**改法**:**不改名**(`a` 在循环里出现 20 次,改动面太大,不满足"一眼看出")。在 `1712`
上方加注释:

```python
# predicted_action 是策略原始输出的冻结副本,只喂给 shadow / candidate 头和 evidence;
# 下面的 a 是仍在被改写的在途命令(close offset → 夹爪控制器 → gripper_only)。
```

这是全函数最容易读错的地方,而修法是零代码。

### 4.2 `record_so101_pv_ee.py:1558-1690` — 加分节注释,不抽函数

**问题**:130 行搭建 source / robot / kin / pv / controller / teleop / dataset。

**改法**:**不抽**。这段的顺序本身是有意义的(注释已写明 OAK 必须先于
`robot.connect()` 启动),抽出去会让这个约束更难看见。只插三行分节注释:

```python
# --- 1. 设备:OAK、机械臂、PV 压力源 ---
# --- 2. 控制器:PV runtime → 夹爪适配器 → EE 控制器 → teleop ---
# --- 3. 数据集:特征 schema 校验 → create/resume → 映射契约 ---
```

### 4.3 `record_so101_pv_ee.py:1651` — `evidence._write_manifest()` 被外部调用

**问题**:`run_recording` 直接调了 `EvidenceSession` 的私有方法,因为它前面刚往
`evidence.manifest` 塞了六个字段。

**改法**:本批次只加一行注释说明"上面这些字段是运行时才知道的,所以要重写一次 manifest"。
真要修就是给 `EvidenceSession` 加一个公开的 `update_manifest(**fields)`——那是独立的
一次改动,不混进这个 PR。

---

## 批次 5a — 文案与入参校验(无闸门)

### 5a.1 `deploy_so101_grip_ee.py:2004` — 提示了不存在的按键

**问题**:`"Press 't' at the lift next time"`,但全文件 `ord(...)` 只有
`[ ] q a s d f w c`,没有 `t`;正确的键是 `a`(`505`)。这条消息恰好在操作员刚丢掉本轮
唯一标签时打印,指错键等于保证下一轮也丢。

**改法**:`'t'` → `'a'`。

### 5a.2 `deploy_so101_grip_ee.py:1300` — 守卫理由与实际取值路径不符

**问题**:`--gripper-telemetry-hz` 的 `ap.error` 文案说边界是
"read back off the bus, not inferred from the command",但每个边界都取自 `actual_pos`,
来源是 `1730` 的 `obs["gripper.pos"]`(`get_observation()`),不是这个 flag 控制的
telemetry reader。守卫本身无害,但理由会误导下一个人。

**改法**:改文案,说明真实理由(telemetry 是事后复核边界所需的 Present_Load / 位置滞后
证据),或者如果确认不需要就删掉这条守卫——**先确认再删**。

### 5a.3 `deploy_so101_grip_ee.py:1305` — stall 计时参数无 argparse 校验

**问题**:`1298` 已经用 `ap.error` 拒了非正的 `--loosen-step` / `--loosen-interval-s`,
但紧挨着的 `--stall-tighten-interval-s`、`--stall-window-s`、`--stall-epsilon` 没有。
传 `--stall-tighten-interval-s 0` 能通过校验、连上机械臂并上电,然后才在 `1554` 的
`TightenRampConfig.__post_init__` 抛裸 `ValueError`——一次 argparse 本可拦下的打字错误,
代价是机械臂已经通电。

**改法**:在 `1305` 的 `if args.stall_tighten_step:` 块里补三条 `ap.error`,与 `1298`
同形。

### 5a.4 `deploy_so101_grip_ee.py:1295` — `--paired-boundaries` 未与 `--correction-dataset-root` 互斥

**问题**:`1735` 的夹爪 elif 链把 `takeover and bool(pv_valid[0])` 放在最前,PV takeover
期间 `1756` 的 `paired.update()` 永不调用。若协议停在 `loosening` 相,`freeze_body` 保持
`True`,机体钉在 `last_predicted_action`,`1901` 的 `paused_cycle` 不断
`t_end += cycle_elapsed_s` 使时长上限永不触发,操作员的 `D`/`F`/`W` 无人消费。只能按 `Q`
结束,该 trial 数据全失。上面四行(`1292`)对 grip candidate 已经拒了完全相同的组合。

**改法**:在 `1295` 的 `if args.paired_boundaries:` 块里补:

```python
if args.correction_dataset_root is not None:
    ap.error("--paired-boundaries cannot be combined with PV correction recording")
```

---

## 批次 5b — 退出路径与按键语义(需物理闸门)

这四条都会改变已验证路径上的运行时行为,单独 PR,闸门结果写进 `docs/RELEASE_AUDIT.md`。

### 5b.1 `deploy_so101_grip_ee.py:2027` — 逃生口会漏掉断连并留着力矩 【高】

**问题**:`relax_all_joints(robot)` 在 `finally` 里调用,内部打印
"Ctrl-C within 3s if you are not holding it" 然后 `time.sleep(3)`。操作员真按了,
`KeyboardInterrupt` 从 `finally` 里穿出去,跳过 `2029` 的 `robot.disconnect()`:
关节没松、串口没关,机械臂继续通电——正是这个函数注释里说要防的 wrist_roll 情形
(第四次那个序列),而且下一次运行开不了口。

**改法**:把倒计时包起来,让中断只跳过 relax、不跳过 disconnect。

```python
if args.relax_on_exit:
    try:
        relax_all_joints(robot)
    except KeyboardInterrupt:
        print("[deploy] relax cancelled by operator; torque held.")
robot.disconnect()
```

**闸门**:跑一次 `--arm-enabled --max-steps 0`,在倒计时里按 Ctrl-C,确认串口关闭
(`ls /dev/serial/by-id/` 后能立刻重连)且力矩保持。

### 5b.2 `deploy_so101_grip_ee.py:1646` — 无关运行里按 `q` 现在会让机械臂瘫软 【高】

**问题**:循环停止条件(`1644-1650`)纳入了 `grip_intervention.stop`(由 `375` 的
`q`/Esc 置位),于是 grip-intervention 场景按 `q` 会一路走到 `2027` 的 relax,机械臂松手
掉物。这是既有工作流的 stop 语义被 stall-ramp 那次改动顺带改掉了:以前 `q` 是带力矩断连
("so the arm keeps its pose",`2028` 注释)。`357` 的窗口图例仍写 `q stop`,没提瘫软,
而 3 秒警告打在操作员没看的终端上。

**改法**(二选一,倾向前者):

1. 从 `1646` 的停止条件里去掉 `grip_intervention.stop`,恢复原语义;或
2. 保留,但把警告画进 cv2 窗口,并把 `357` 的图例改成 `q stop (arm relaxes)`。

**闸门**:intervention 场景下夹住物体按 `q`,确认物体不掉。

### 5b.3 `deploy_so101_grip_ee.py:1570` + `505` — `A` 被重载,中止失败的 ramp 会写出伪造的 `lift_boundary` 【中】

**问题**:`1570` 的横幅写 "Press 'A' again to hand back to ACT",`TightenRampOperator`
把 `a` 当普通 toggle;但 `StallTightenRamp.update(engaged=False)` 只要 `steps_applied`
非零就记 `self.lift_boundary = actual_pos`。操作员收紧、发现物体还是起不来、按提示交还
控制权——就往 evidence 里塞了一个与真实边界无法区分的值。`PairedBoundaryProtocol` 靠拆成
`A`(收紧)/ `S`(已抬起)避开了这个问题,独立路径没有。

**改法**:给独立路径也拆键,与 paired 路径对齐——`A` 进入/退出收紧,`S` 确认"已抬起"
并记边界;未按 `S` 就退出时,记 `lift_boundary = None` 并打印一行说明。改动落在 `461-514`
的 `TightenRampOperator` 与 `StallTightenRamp.update` 的 disengage 分支。

**闸门**:一次成功抬起(按 `S`)与一次放弃(只按 `A`)各跑一遍,核对
`stall_tighten_result.lift_boundary` 分别为数值与 `null`。

### 5b.4 `deploy_so101_grip_ee.py:1756` — 冻结期两个分支返回过期的 ACT 夹爪目标 【中】

**问题**:`freeze_body` 期间 `a = last_predicted_action.copy()`(`1681`)且策略不重跑,
所以交给 `paired.update` 的 `policy_target` 是**抬起之前**的 ACT 夹爪指令。
`PairedBoundaryProtocol.update` 在 `drop_marked` 和 `lift_unconfirmed` 两个分支原样返回它,
于是命令在一个周期内从松开 ramp 的位置跳回 ACT 的抓取值——一次全程闭合。`W` 路径上物体
还在爪里,这与 `617` 给操作员的提示("the jaw stays where it is")直接矛盾。

**改法**:这两个分支返回 `self._target`(或 `actual_pos`),兑现已声明的契约。
改动在 `packages/` 侧的 `PairedBoundaryProtocol.update`,不在本文件。

**闸门**:夹住物体、进入松开 ramp、按 `F`(掉落)与 `W`(撤销)各一次,确认夹爪不跳变。

---

## 每批次的验证

```bash
# 全量(1025 tests)——批次 1–4 结束后必须全绿,且不新增/修改测试
env -u PYTHONPATH ../.venv-lerobot/bin/python -m pytest -q

# 抽函数后必跑的四个守卫(批次 3)
env -u PYTHONPATH ../.venv-lerobot/bin/python -m pytest -q \
  packages/so101_teleop/tests/test_every_python_file_compiles.py \
  packages/so101_teleop/tests/test_programs_import_cleanly.py \
  packages/so101_teleop/tests/test_programs_only_use_existing_api.py \
  packages/so101_teleop/tests/test_no_module_comes_from_outside_this_repo.py

# 不接机械臂的干跑(批次 1–3 之后各跑一次)
env -u PYTHONPATH ../.venv-lerobot/bin/python \
  integrations/pressurevision/tools/record_so101_pv_ee.py --check-config …
env -u PYTHONPATH ../.venv-lerobot/bin/python \
  integrations/pressurevision/tools/record_so101_pv_ee.py --stream-preflight …
env -u PYTHONPATH ../.venv-lerobot/bin/python \
  integrations/pressurevision/tools/deploy_so101_grip_ee.py --max-steps 0 …   # 不带 --arm-enabled
```

批次 1–4 的判据是**测试集一行不改而全绿**。如果某处改动需要动测试,说明它不是纯搬运,
退回去重做或移到 5b。

---

## 明确不做

- **不动 `deploy` 主循环体(1651-1907)**。它是 `a` 的顺序改写加一条 elif 链;任何抽取都会
  把"谁改了 `a[gripper_index]`"这个唯一真相分散到两个地方,不满足"一眼看出不改变行为"。
  它的可读性问题用 4.1 的注释解决。
- **不动 `ee_controller.py` / `ee_control.py`**。
- **不动 `record` 的 `1558-1690` 搭建段**(见 4.2)。
- **不重命名 `deploy` 循环里的 `a`**(见 4.1)。
- **不统一 5b 之外的任何标签/schema**。3.5 只是把不一致暴露出来,不消除它。
