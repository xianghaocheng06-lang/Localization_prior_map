"""Select focus anchors, bracket captures per board pose, fit and validate a focus map."""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import cv2
import numpy as np

from calibration.camera_calibration import (autofocus_mode, board_spec, calibrate_images, detect_corners,
    object_points, read_image, read_json, verify_focus, write_image, write_json,
    set_manual_focus, FocusControlError)
from localization.focus_map import FocusMap

ROOT = Path(__file__).resolve().parents[1]


class StopCapture(Exception):
    pass


def position_key(q):
    return f"focus_{q:g}".replace('.', 'p').replace('-', 'm')


def focus_list(text):
    values = sorted(float(x) for x in text.split(','))
    if len(values) < 2 or not np.isfinite(values).all() or min(values)<0 or len(set(values)) != len(values):
        raise ValueError('至少指定两个不同的有限焦点值，例如 --focus-values 200,260,320；示例值并非你的推荐值')
    return values


def image_for_capture(cap, settings):
    ok, image = cap.read()
    if not ok:
        raise ValueError('相机读帧失败')
    if list(image.shape[1::-1]) != settings['image_size']:
        raise ValueError(f'实际分辨率 {image.shape[1::-1]} 与设置不同')
    return cv2.flip(image, -1) if settings['flip_camera_180'] else image


def show(image, lines, corners=None, pattern=(9, 6)):
    preview = image.copy()
    if corners is not None:
        cv2.drawChessboardCorners(preview, tuple(pattern), corners, True)
    for i, line in enumerate(lines):
        cv2.putText(preview, line, (8, 22+22*i), cv2.FONT_HERSHEY_SIMPLEX, .46, (0, 225, 255), 1, cv2.LINE_AA)
    cv2.imshow('Autofocus calibration', preview)
    key = cv2.waitKey(1) & 0xff
    if key in (27, ord('q'), ord('Q')) or cv2.getWindowProperty('Autofocus calibration', cv2.WND_PROP_VISIBLE) < 1:
        raise StopCapture
    return key


def stabilized_frame(cap, q, args, settings, status, on_control=None):
    """Wait with live preview; require several stable detections after lens settling."""
    def preview_control(image, phase, state):
        if list(image.shape[1::-1]) != settings['image_size']:
            raise ValueError('控制期间相机图像尺寸不符')
        if settings['flip_camera_180']:
            image = cv2.flip(image,-1)
        show(image,[status,phase,f"requested={q:g} actual={state['focus_readback']} mode={state['autofocus_mode']}"])
    set_manual_focus(cap,q,settle=args.settle,timeout=args.timeout,
                     on_frame=preview_control,on_event=on_control)
    started = time.monotonic()
    previous = None
    stable = 0
    while time.monotonic()-started < args.timeout:
        image = image_for_capture(cap, settings)
        elapsed = time.monotonic()-started
        state = verify_focus(cap, q)
        corners = detect_corners(image, (args.cols, args.rows))
        sharpness = 0.
        if corners is not None:
            x, y, w, h = cv2.boundingRect(corners)
            gray = cv2.cvtColor(image[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
            sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            # Corner numbering can reverse by 180 degrees.
            motion = float('inf') if previous is None else min(
                np.sqrt(np.mean(np.sum((corners.reshape(-1, 2)-p)**2, axis=1)))
                for p in [previous.reshape(-1, 2), previous.reshape(-1, 2)[::-1]])
            stable = stable+1 if motion <= args.max_motion_px and sharpness >= args.min_sharpness else 0
        else:
            stable = 0
        previous = corners
        show(image, [status, f'focus={q:g} board={corners is not None} sharpness={sharpness:.0f}',
                     f'hold still: {stable}/3; Q: finish without fitting'], corners, (args.cols, args.rows))
        if stable >= 3:
            verify_focus(cap, q)
            return image, state, sharpness
    return None


def capture(args):
    cfg = read_json(args.config)
    settings = dict(camera_index=cfg['camera_index'] if args.camera is None else args.camera,
        image_size=cfg['camera_resolution'], camera_fps=cfg['camera_fps'], flip_camera_180=cfg['flip_camera_180'])
    spec = board_spec(args.cols, args.rows, args.square_mm)
    session = args.output / datetime.now().strftime('session_%Y%m%d_%H%M%S_%f')
    session.mkdir(parents=True)
    backend_id = {'dshow': cv2.CAP_DSHOW, 'msmf': cv2.CAP_MSMF, 'any': cv2.CAP_ANY}[args.backend]
    cap = cv2.VideoCapture(settings['camera_index'], backend_id)
    entries, events = {}, []
    anchors = focus_list(args.focus_values) if args.focus_values else []
    slots = {}
    pose_number = 0
    finish_fit = False
    session_data = {**settings, **spec, 'focus_values': anchors, 'groups': [], 'mode': 'select',
                    'validation_every': args.validation_every, 'created_at': datetime.now().astimezone().isoformat()}

    def save_group_metadata(folder, q):
        if folder not in entries:
            entries[folder] = {**settings, **spec, 'backend': cap.getBackendName(), 'images_already_oriented': True,
                'focus_position': q, 'focus_lock_verified': True, 'focus_samples': {}}
        return entries[folder]

    try:
        if not cap.isOpened():
            raise ValueError('无法打开相机，先关闭其他相机程序')
        width, height = settings['image_size']
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, settings['camera_fps'])
        cv2.namedWindow('Autofocus calibration', cv2.WINDOW_NORMAL)
        print(f'会话目录：{session.resolve()}\n选择阶段：A自动对焦；在实际近/中/远距离清晰后，按1/2/3记录；Enter确认。\n'
              '采集阶段：每次换棋盘姿态，按S/空格，持稳等待自动切换所有焦点。每第5姿态默认用于验证。\n'
              '建议15-20种姿态；F结束并标定，Q只保存后退出。', flush=True)
        selecting = not anchors
        while True:
            image = image_for_capture(cap, settings)
            raw_focus = float(cap.get(cv2.CAP_PROP_FOCUS))
            raw_af = float(cap.get(cv2.CAP_PROP_AUTOFOCUS))
            if selecting:
                key = show(image, [f'SELECT: focus={raw_focus:g} AF={autofocus_mode(raw_af,cap.getBackendName())}',
                    'A:autofocus 1/2/3:record near/middle/far focus Enter:start',
                    f'slots: {slots}; use real working distances; wait until clear'])
                if key in (ord('a'), ord('A')):
                    if not cap.set(cv2.CAP_PROP_AUTOFOCUS, 1):
                        raise ValueError('设备拒绝自动对焦')
                elif key in (ord('1'), ord('2'), ord('3')):
                    if not np.isfinite(raw_focus) or raw_focus < 0:
                        print('焦点回读无效，不能记录', flush=True)
                    else:
                        slots[chr(key)] = raw_focus
                        print(f'记录档位 {chr(key)} = {raw_focus:g}；请人工确认清晰且读数已稳定', flush=True)
                elif key in (10, 13):
                    anchors = sorted(set(slots.values()))
                    if len(anchors) < 2:
                        print('至少记录两个不同焦点；建议记录实际近/中/远三个', flush=True)
                    else:
                        selecting = False
                continue
            if session_data['mode'] != 'capture':
                session_data.update(focus_values=anchors, mode='capture')
                write_json(session/'session.json', session_data)
            counts = [len(entries.get(session/'train'/position_key(q), {}).get('focus_samples', {})) for q in anchors]
            key = show(image, [f'CAPTURE anchors={anchors} poses={pose_number}',
                f'train counts={counts}; need >=10 each; change pose between sweeps',
                'S/Space:sweep F:finish and fit Q:save and quit'])
            if key in (ord('f'), ord('F')):
                finish_fit = True
                break
            if key not in (ord('s'), ord('S'), 32):
                continue
            pose_number += 1
            validation = pose_number % args.validation_every == 0
            targets = list(anchors)
            if validation:
                # These intermediate positions are never used to fit anchor intrinsics.
                targets += [float(round((a+b)/2)) for a,b in zip(anchors, anchors[1:])]
            group = {'pose_id': pose_number, 'split': 'validation' if validation else 'train', 'results': []}
            session_data['groups'].append(group)
            for q in sorted(set(targets)):
                status = f'POSE {pose_number}: {group["split"]}; HOLD BOARD STILL'
                def control_event(event):
                    events.append({'pose':pose_number,'kind':'focus_control',**event})
                    write_json(session/'events.json',events)
                    if event['action'] in ['write_focus','focus_timeout','verified']:
                        print(f"焦点{q:g} {event['action']}：回读 {event['readback']}",flush=True)
                control_error = None
                try:
                    value = stabilized_frame(cap, q, args, settings, status,on_control=control_event)
                except FocusControlError as error:
                    value, control_error = None, str(error)
                record = {'focus_position': q, 'saved': value is not None}
                if control_error:
                    record['error']=control_error
                group['results'].append(record)
                events.append({'time': datetime.now().astimezone().isoformat(), 'pose': pose_number,
                               'focus': q, 'success': value is not None})
                write_json(session/'events.json', events)
                if value is None:
                    reason = control_error or '超时/模糊/未检出/未持稳；可到该焦点清晰的距离补拍'
                    print(f'姿态{pose_number} 焦点{q:g}：跳过，{reason}', flush=True)
                    continue
                frame, state, sharpness = value
                folder = session/group['split']/position_key(q)
                metadata = save_group_metadata(folder, q)
                name = f'pose_{pose_number:04d}.png'
                write_image(folder/name, frame)
                metadata['focus_samples'][name] = {**state, 'sharpness': sharpness}
                write_json(folder/'capture.json', metadata)
                record.update(image=str((folder/name).relative_to(session)), sharpness=sharpness)
                print(f'姿态{pose_number} 焦点{q:g}：已保存 {group["split"]}', flush=True)
            write_json(session/'session.json', session_data)
    except StopCapture:
        pass
    finally:
        session_data['focus_values'] = anchors
        write_json(session/'session.json', session_data)
        cap.release()
        cv2.destroyAllWindows()
    print(f'采集已保存：{session.resolve()}\n复算命令：python autofocus_calibration.py fit --session "{session.resolve()}"', flush=True)
    if finish_fit:
        fit(SimpleNamespace(session=session, overwrite=False, max_validation_rms=args.max_validation_rms))


def heldout_error(folder, K, D, spec):
    errors = []
    obj = object_points(spec)
    for path in sorted(folder.glob('*.png')):
        corners = detect_corners(read_image(path), spec['inner_corners'])
        if corners is None:
            continue
        ok, rv, tv = cv2.solvePnP(obj, corners, K, D, flags=cv2.SOLVEPNP_ITERATIVE)
        if ok:
            projected = cv2.projectPoints(obj,rv,tv,K,D)[0]
            errors.append(float(np.sqrt(np.mean(np.sum((projected-corners)**2,axis=2)))))
    return errors


def interval_is_valid(errors, threshold):
    return bool(len(errors) >= 3 and np.isfinite(errors).all() and np.max(errors) <= threshold)


def fit(args):
    session = args.session.resolve()
    data = read_json(session/'session.json')
    anchors = data['focus_values']
    if len(anchors) < 2:
        raise ValueError('至少需要两档不同焦点')
    output = session/'calibrated'
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise ValueError('标定输出已存在；复算请加 --overwrite')
    # Fail early: fewer than 10 good captures per anchor cannot be fit reliably by this workflow.
    for q in anchors:
        folder = session/'train'/position_key(q)
        if len(list(folder.glob('*.png'))) < 10:
            raise ValueError(f'焦点{q:g}训练图不足10张，请补采新会话或增加姿态；每第5姿态默认留作验证')
    output.mkdir(parents=True, exist_ok=True)
    entries = []
    for q in anchors:
        path = output/f'{position_key(q)}.json'
        calibrate_images(SimpleNamespace(images=session/'train'/position_key(q), output=path,
            cols=None, rows=None, square_mm=None, flip_180=None, max_view_error=None, overwrite=args.overwrite))
        entries.append(read_json(path))
    mapping = {'model': 'opencv_pinhole_focus_map', 'image_size': data['image_size'],
        'flip_camera_180': data['flip_camera_180'], 'camera_index': data['camera_index'],
        'created_at': datetime.now().astimezone().isoformat(), 'entries': entries,
        'validated_intervals': [], 'validation_max_rms_px': args.max_validation_rms,
        'focus_control_backend': entries[0].get('focus_control_backend'),
        'note': 'Linear parameter interpolation. Held-out reprojection checks do not prove metric pose accuracy.'}
    map_path = output/'focus_map.json'
    # Do not publish a new final map until validation has completed.
    pending_path = output/'focus_map.pending.json'
    write_json(pending_path, mapping)
    table = FocusMap(pending_path, data['image_size'], data['flip_camera_180'])
    validations = []
    for q in anchors:
        K,D = table.lookup(q)
        errors = heldout_error(session/'validation'/position_key(q),K,D,data)
        validations.append({'focus': q, 'kind': 'anchor', 'errors_px': errors,
                            'passed': interval_is_valid(errors,args.max_validation_rms)})
    for low,high in zip(anchors,anchors[1:]):
        q = float(round((low+high)/2))
        if q <= low or q >= high:
            validations.append({'focus': q, 'kind': 'midpoint', 'errors_px': [], 'passed': False})
            continue
        K,D = table.lookup(q, require_validation=False)
        errors = heldout_error(session/'validation'/position_key(q),K,D,data)
        # Also require endpoint holdouts to pass, not just a good middle fit.
        endpoints_ok = all(next(v['passed'] for v in validations if v['kind']=='anchor' and v['focus']==x) for x in [low,high])
        passed = endpoints_ok and interval_is_valid(errors,args.max_validation_rms)
        validations.append({'focus': q, 'kind': 'midpoint', 'errors_px': errors, 'passed': passed})
        if passed:
            mapping['validated_intervals'].append([low,high])
    mapping['validation'] = validations
    write_json(map_path,mapping)
    pending_path.unlink(missing_ok=True)
    rows = ['# 多焦点标定结果', '', '每个完整档位独立标定；验证姿态完全未参与训练，区间还检查了未参与拟合的中间焦点。', '',
        '| 焦点 | 训练图 | 训练 RMS px |', '|---|---:|---:|']
    for entry in entries:
        rows.append(f"| {entry['focus_position']:g} | {entry['num_images']} | {entry['rms']:.4f} |")
    rows += ['', '| 验证焦点 | 类型 | 张数 | 最大 RMSE px | 通过 |', '|---|---|---:|---:|---|']
    for v in validations:
        maximum = f"{max(v['errors_px']):.4f}" if v['errors_px'] else '无数据'
        rows.append(f"| {v['focus']:g} | {v['kind']} | {len(v['errors_px'])} | {maximum} | {v['passed']} |")
    rows += ['', f"已通过的插值区间：{mapping['validated_intervals']}。至少3张独立姿态且每张重投影RMSE≤{args.max_validation_rms:g}px才通过。",
        '以上结果来自独立棋盘图的重投影检查。档位间的焦距、主点和畸变系数均按线性插值计算。',
        '自动对焦时普通驱动回读不保证与曝光帧严格同步；运行时丢弃调焦读数变化的画面并等待连续稳定读数，仍需实机验证。', '',
        '定位启动示例（项目根目录）：', '```powershell',
        f'python positioning.py --camera {data["camera_index"]} --camera-backend dshow --focus-map "{map_path}" --csv positioning_dynamic_focus.csv', '```']
    (output/'标定报告.md').write_text('\n'.join(rows)+'\n',encoding='utf-8')
    print(f'焦点表：{map_path}\n报告：{output / "标定报告.md"}\n已验证区间：{mapping["validated_intervals"]}',flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command',required=True)
    c = commands.add_parser('capture',help='选择近/中/远焦点，按姿态自动切焦采图，F批量标定')
    c.add_argument('--config',type=Path,default=ROOT/'config/positioning_config.json')
    c.add_argument('--camera',type=int)
    c.add_argument('--backend',choices=['dshow','msmf','any'],default='dshow')
    c.add_argument('--focus-values',help='可选：已知的至少两个焦点，用逗号分隔；不填则在窗口中选择')
    c.add_argument('--cols',type=int,default=9)
    c.add_argument('--rows',type=int,default=6)
    c.add_argument('--square-mm',type=float,default=20.)
    c.add_argument('--settle',type=float,default=1.5)
    c.add_argument('--timeout',type=float,default=2.5)
    c.add_argument('--min-sharpness',type=float,default=50.)
    c.add_argument('--max-motion-px',type=float,default=.5)
    c.add_argument('--validation-every',type=int,default=5)
    c.add_argument('--output',type=Path,default=ROOT/'calibration/focus_maps')
    c.add_argument('--max-validation-rms',type=float,default=1.)
    f = commands.add_parser('fit',help='重算已保存的多焦点会话')
    f.add_argument('--session',type=Path,required=True)
    f.add_argument('--overwrite',action='store_true')
    f.add_argument('--max-validation-rms',type=float,default=1.)
    args = parser.parse_args()
    try:
        if not np.isfinite(args.max_validation_rms) or args.max_validation_rms <= 0:
            raise ValueError('验证误差阈值须为有限正数')
        if args.command == 'capture':
            for key in ['settle','timeout','min_sharpness','max_motion_px']:
                if not np.isfinite(getattr(args,key)) or getattr(args,key)<0:
                    raise ValueError(f'{key} 须为有限非负数')
            if args.timeout <= 0 or args.validation_every < 2:
                raise ValueError('timeout 须为正，validation-every 至少为2')
            capture(args)
        else:
            fit(args)
    except (OSError,ValueError,KeyError,cv2.error) as error:
        print(f'错误：{error}',file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
