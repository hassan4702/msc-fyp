# Frontend

React app (to be scaffolded with Vite in Phase 4):

- Webcam capture via `getUserMedia` — frames are sampled client-side and sent to
  `POST /chat`. **Raw video never leaves the browser persistently** (GDPR).
- Chat window.
- Live emotion indicator driven by the `text_emotion` / `face_emotion` /
  `fused_emotion` fields returned by the API.

Create with:

```bash
npm create vite@latest frontend -- --template react-ts
```
