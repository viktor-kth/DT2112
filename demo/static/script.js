/*
    The inspiration for this script comes from the following tutorial:
    https://www.youtube.com/watch?v=qs8FQ_oT33g
*/

const recordButton = window.document.getElementById('record');
const recordText = window.document.getElementById('record-text');
const output = window.document.getElementById('output');
const visualizer = document.getElementById('visualizer');
const canvasCtx = visualizer.getContext('2d');
const celebrityList = document.getElementById('celebrityList');
let audioContext, analyser, source;

if (navigator.mediaDevices.getUserMedia) {
    console.log('getUserMedia supported.');

    let onMediaSetupSuccess = function(stream) {
        console.log('MediaRecorder started.');
        const mediaRecorder = new MediaRecorder(stream);
        let audioChunks = [];
        
        audioContext = new AudioContext();
        analyser = audioContext.createAnalyser();
        source = audioContext.createMediaStreamSource(stream);
        source.connect(analyser);
        analyser.fftSize = 256;
        const bufferLength = analyser.frequencyBinCount;
        const dataArray = new Uint8Array(bufferLength);
        
        function draw() {
           
            analyser.getByteFrequencyData(dataArray);
            canvasCtx.beginPath();
            canvasCtx.clearRect(0, 0, visualizer.width, visualizer.height);

            const centerX = visualizer.width / 2;
            const centerY = visualizer.height / 2;
            let radian = 0;
            let totalLength = bufferLength/3;
            let radianAdd = (Math.PI * 2) * (1.0/totalLength);
            const radius = 70;
          


            if( mediaRecorder.state === 'recording') {
                for (let i = 0; i < totalLength; i++) {
                let value = dataArray[i]  / 3;

                    let x = centerX + Math.cos(radian) * radius;
                    let y = centerY + Math.sin(radian) * radius/2;
                    canvasCtx.arc(x, y, value, radian, radian + radianAdd);
                    canvasCtx.stroke();
                    radian += radianAdd;
                
                }
            }
            
            requestAnimationFrame(draw);
        }
        draw();

        recordButton.addEventListener('click', () => {
            if (mediaRecorder.state === 'recording') {
                mediaRecorder.stop();
                recordText.innerHTML = 'Record';
            } else {
                mediaRecorder.start();
                recordText.innerHTML = 'Stop';
            }
        });

        mediaRecorder.ondataavailable = function(event) {
            if (event.data.size > 0) {
                audioChunks.push(event.data);
            }
        };

        mediaRecorder.onstop = function() {
            let audioBlob = new Blob(audioChunks, {type: 'audio/wav'});
            audioChunks = [];

            let formData = new FormData();
            formData.append('audio',audioBlob);

            fetch('/findmatch', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                output.innerHTML = data.message;
            });
        }
    }

    let onMediaSetupError = function(error) {
        console.log('MediaRecorder error: ', error);
    }
    
    navigator.mediaDevices
    .getUserMedia({audio: true})
    .then(onMediaSetupSuccess)
    .catch(onMediaSetupError);
} else {
    alert('getUserMedia not supported.');
}
