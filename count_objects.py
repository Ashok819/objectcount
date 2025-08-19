import argparse, csv, time
from pathlib import Path
import cv2
from ultralytics import YOLO

def count_per_class(results, conf_thresh=0.25):
    counts = {}
    names = results.names
    for r in results:
        for box, conf, cls in zip(r.boxes.xyxy.cpu().numpy(),
                                  r.boxes.conf.cpu().numpy(),
                                  r.boxes.cls.cpu().numpy()):
            if conf < conf_thresh: 
                continue
            label = names[int(cls)]
            counts[label] = counts.get(label, 0) + 1
    counts["__total"] = sum(v for k,v in counts.items() if k!="__total")
    return counts

def annotate_frame(frame, r, names, conf_thresh=0.25):
    for box, conf, cls in zip(r.boxes.xyxy.cpu().numpy(),
                              r.boxes.conf.cpu().numpy(),
                              r.boxes.cls.cpu().numpy()):
        if conf < conf_thresh: 
            continue
        x1,y1,x2,y2 = map(int, box)
        cv2.rectangle(frame, (x1,y1), (x2,y2), (255,255,255), 2)
        label = f"{names[int(cls)]} {conf*100:.0f}%"
        t_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1-18), (x1 + t_size[0] + 6, y1), (255,255,255), -1)
        cv2.putText(frame, label, (x1+3, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
    return frame

def save_counts_csv(counts, out_csv):
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class","count"])
        w.writerow(["total", counts.get("__total", 0)])
        for k,v in sorted((k,v) for k,v in counts.items() if k!="__total"):
            w.writerow([k,v])

def main():
    p = argparse.ArgumentParser(description="Object counting with YOLOv8")
    p.add_argument("--source", required=True, help="image/video path or 0 for webcam")
    p.add_argument("--model", default="yolov8n.pt", help="YOLOv8 model (n/s/m/l/x or a path)")
    p.add_argument("--conf", type=float, default=0.25, help="confidence threshold")
    p.add_argument("--save", action="store_true", help="save annotated video/image")
    p.add_argument("--csv", default=None, help="path to save CSV counts")
    args = p.parse_args()

    model = YOLO(args.model)

    # Single image
    if str(args.source).lower().endswith((".jpg",".jpeg",".png",".bmp",".webp",".tif",".tiff")):
        results = model(args.source, conf=args.conf, verbose=False)
        counts = count_per_class(results, conf_thresh=args.conf)
        print("Counts:", counts)
        if args.csv:
            save_counts_csv(counts, args.csv)
            print("Saved CSV:", args.csv)
        if args.save:
            img = cv2.imread(args.source)
            r0 = results[0]
            annotated = annotate_frame(img, r0, results[0].names, conf_thresh=args.conf)
            out_path = str(Path(args.source).with_stem(Path(args.source).stem + "_annotated"))
            cv2.imwrite(out_path, annotated)
            print("Saved image:", out_path)
        return

    # Video or webcam
    cap = cv2.VideoCapture(0 if args.source == "0" else args.source)
    if not cap.isOpened():
        raise SystemExit("Could not open source")

    # Prepare writer if saving
    writer = None
    if args.save:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_path = "annotated.mp4"
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(out_path, fourcc, fps, (w,h))
        print("Saving to", out_path)

    window = "Object Counter"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    last_counts = {}
    last_csv_time = 0
    while True:
        ok, frame = cap.read()
        if not ok: break
        results = model.predict(frame, conf=args.conf, verbose=False)
        r0 = results[0]
        frame = annotate_frame(frame, r0, r0.names, conf_thresh=args.conf)

        # live counts
        counts = count_per_class(results, conf_thresh=args.conf)
        last_counts = counts

        # overlay summary
        y = 24
        cv2.rectangle(frame, (10,10), (260, 10 + 18*(min(10, len(counts))+1)), (255,255,255), -1)
        cv2.putText(frame, f"Total: {counts.get('__total',0)}", (16,y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2); y+=20
        for k,v in sorted((k,v) for k,v in counts.items() if k!="__total"):
            cv2.putText(frame, f"{k}: {v}", (16,y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1); y+=18

        if writer: writer.write(frame)
        cv2.imshow(window, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC to quit
            break
        # press 'c' every ~2s to save CSV snapshot if --csv given as a path
        if args.csv and time.time() - last_csv_time > 2:
            save_counts_csv(counts, args.csv)
            last_csv_time = time.time()

    cap.release()
    if writer: writer.release()
    cv2.destroyAllWindows()
    if last_counts and args.csv:
        print("Final counts:", last_counts)
        print("Saved CSV:", args.csv)

if __name__ == "__main__":
    main()
