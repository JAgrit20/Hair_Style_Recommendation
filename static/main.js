var video = document.getElementById('video');
// getUserMedia()Get camera footage with
var media = navigator.mediaDevices.getUserMedia({ video: true });
//Pour into video tags for real-time playback (streaming)
media.then((stream) => {
    video.srcObject = stream;
});

var canvas = document.getElementById('canvas');
canvas.setAttribute('width', video.width);
canvas.setAttribute('height', video.height);

video.addEventListener(
    'timeupdate',
    function () {
        var context = canvas.getContext('2d');
        context.drawImage(video, 0, 0, video.width, video.height);
    },
    true
);

//Set the listener that executes capture acquisition when the space key is pressed
document.addEventListener('keydown', (event) => {
    var keyName = event.key;
    if (keyName === '1') {
        console.log(`keydown: SpaceKey`);
        context = canvas.getContext('2d');
        //Remove the head of the acquired base64 data
        var img_base64 = canvas.toDataURL('image/jpeg').replace(/^.*,/, '')
        captureImg(img_base64);
    }
});

var xhr = new XMLHttpRequest();

//Captured image data(base64)POST
function captureImg(img_base64) {
    const body = new FormData();
    body.append('img', img_base64);
    xhr.open('POST', 'http://localhost:8000/capture_img', true);
    xhr.onload = () => {
        console.log(xhr.responseText)
    };
    xhr.send(body);
}
