/**
 * 指定されたURLとIDを持つ要素にWebSocketを介して映像をストリーミングします。
 * @param {string} url - 接続先のWebSocket URL
 * @param {string} elementId - 映像を表示する<img>タグのID
 */
function startVideoStream(url, elementId) {
  const ws = new WebSocket(url);
  const img = document.getElementById(elementId);

  if (!img) {
    console.error(`Error: Element with ID '${elementId}' not found.`);
    return;
  }

  ws.onmessage = function (event) {
    const blob = new Blob([event.data], { type: 'image/jpeg' });
    const objectURL = URL.createObjectURL(blob);
    img.src = objectURL;
  };

  ws.onclose = function (event) {
    console.log(`WebSocket for ${elementId} closed.`);
  };

  ws.onerror = function (error) {
    console.error(`WebSocket Error for ${elementId}: `, error);
  };
}