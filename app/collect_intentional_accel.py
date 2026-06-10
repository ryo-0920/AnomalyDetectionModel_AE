import carla
import random
import time
import cv2
import numpy as np
import os
import csv
import math

def main():
    client = carla.Client("localhost", 2000)
    client.set_timeout(10.0)

    # ==== ワールド・自車両 ====
    world = client.load_world("Town01")
    blueprint_library = world.get_blueprint_library()

    vehicle_bp = blueprint_library.filter("model3")[0]
    spawn_points = world.get_map().get_spawn_points()
    spawn_point = random.choice(spawn_points)
    vehicle = world.spawn_actor(vehicle_bp, spawn_point)
    vehicle.set_autopilot(True)

    # ==== NPC車両を複数追加 ====
    npc_vehicles = []
    npc_bp_list = blueprint_library.filter("vehicle.*")
    for i in range(5):
        npc_bp = random.choice(npc_bp_list)
        npc_spawn = random.choice(spawn_points)
        npc = world.try_spawn_actor(npc_bp, npc_spawn)
        if npc is not None:
            npc.set_autopilot(True)
            npc_vehicles.append(npc)
    print(f"NPC 車両 {len(npc_vehicles)} 台をスポーンしました")

    # ==== カメラ ====
    camera_bp = blueprint_library.find("sensor.camera.rgb")
    camera_bp.set_attribute("image_size_x", "800")
    camera_bp.set_attribute("image_size_y", "600")
    camera_bp.set_attribute("fov", "90")
    camera_transform = carla.Transform(carla.Location(x=1.5, z=2.0))
    camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)

    # ==== 出力先 ====
    output_folder = r"C:\Users\user\Desktop\Work\carla\Gofumi\datarecode_train"
    os.makedirs(output_folder, exist_ok=True)
    csv_path = os.path.join(output_folder, "intentional_accel.csv")
    csv_file = open(csv_path, mode="w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["frame", "steer", "throttle", "brake", "speed", "intent", "is_accel"])

    frame_count = 0
    accel_active = False
    accel_start_time = 0.0
    rise_time = 1.5  # 秒

    prev_speed = 0.0
    prev_throttle = 0.0

    # ==== 補助関数 ====
    def get_speed(vehicle):
        v = vehicle.get_velocity()
        return math.sqrt(v.x**2 + v.y**2 + v.z**2)

    def throttle_curve(elapsed_time, rise_time=1.5, base=0.0, max_val=1.0):
        ratio = min(1.0, elapsed_time / rise_time)
        noise = random.uniform(-0.03, 0.03)
        return min(1.0, base + (max_val - base) * ratio + noise)

    # ==== カメラコールバック ====
    def process_img(image):
        nonlocal frame_count, accel_active, accel_start_time
        nonlocal prev_speed, prev_throttle

        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        rgb = array.reshape((image.height, image.width, 4))[:, :, :3].copy()

        control = vehicle.get_control()
        current_speed = get_speed(vehicle)
        current_time = time.time()
        intent = "normal"
        is_accel = 0

        # ---- 急加速発動条件: 低速・加速中・減速していない・ランダム ----
        if (not accel_active
            and current_speed >= prev_speed  # 減速していない
            and current_speed < 15.0         # 低速（発進シーン）
            and control.throttle > prev_throttle):  # アクセル踏み増し
            if random.randint(1,3) == 1:
                accel_active = True
                accel_start_time = current_time
                print(f"⚡ Intentional accel start at frame {frame_count}, speed={current_speed:.2f}")

        # ---- 急加速中 ----
        if accel_active:
            elapsed = current_time - accel_start_time
            if elapsed <= rise_time:
                control.throttle = throttle_curve(elapsed, rise_time)
                control.brake = 0.0
                vehicle.apply_control(control)
                intent = "intentional_accel"
                is_accel = 1
            else:
                accel_active = False

        # ---- CSV書き込み ----
        csv_writer.writerow([
            frame_count,
            control.steer,
            control.throttle,
            control.brake,
            current_speed,
            intent,
            is_accel
        ])

        # ---- 画像保存 ----
        img_path = os.path.join(output_folder, f"frame_{frame_count:06d}.png")
        cv2.imwrite(img_path, rgb)

        # ---- 前フレーム更新 ----
        prev_speed = current_speed
        prev_throttle = control.throttle
        frame_count += 1

    # ==== メインループ ====
    camera.listen(lambda image: process_img(image))
    print("🚗 Running... Ctrl+C to stop")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        camera.stop()
        camera.destroy()
        vehicle.destroy()
        for npc in npc_vehicles:
            npc.destroy()
        csv_file.close()
        print("✅ Cleanup done.")

if __name__ == "__main__":
    main()
