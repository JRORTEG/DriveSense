# Sample footage

No webcam yet, so the CV pipeline is developed against these clips. The videos are gitignored; download them into this folder:

```
curl -L -o ampel_red_to_green.ogv "https://upload.wikimedia.org/wikipedia/commons/b/ba/D%C3%BCsseldorfer_Ampel_von_Rot_nach_Gr%C3%BCn.ogv"
curl -L -o intersection_montreal_720p.webm "https://upload.wikimedia.org/wikipedia/commons/transcoded/1/18/Intersection_Pie-IX-Sherbrooke.webm/Intersection_Pie-IX-Sherbrooke.webm.720p.vp9.webm"
```

| File | Use | Source / license |
|---|---|---|
| `ampel_red_to_green.ogv` | Task 2 (traffic light). 240x320, 15 FPS, red→green at ~8.2 s. Fixed ROI `100,90,55,80`. | [Düsseldorfer Ampel von Rot nach Grün](https://commons.wikimedia.org/wiki/File:D%C3%BCsseldorfer_Ampel_von_Rot_nach_Gr%C3%BCn.ogv) by Noebse — public domain |
| `intersection_montreal_720p.webm` | Tasks 3–6 (vehicles, tracking, ego motion). 1280x720, 30 FPS, handheld. | [Intersection Pie-IX-Sherbrooke](https://commons.wikimedia.org/wiki/File:Intersection_Pie-IX-Sherbrooke.webm) by Thomas1313 — CC BY-SA 4.0 |

Try the detector: `python -m cv.traffic_light data/samples/ampel_red_to_green.ogv --roi 100,90,55,80`
