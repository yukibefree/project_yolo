const ws = new WebSocket("ws://localhost:8000/ws");
const img = document.getElementById("video-feed");

ws.onmessage = function (event) {
  // サーバーから画像データが届くと実行
  const blob = new Blob([event.data], { type: 'image/jpeg' });
  const url = URL.createObjectURL(blob);
  img.src = url;
};

ws.onclose = function (event) {
  console.log("WebSocket connection closed.");
};

ws.onerror = function (error) {
  console.error("WebSocket Error: ", error);
};