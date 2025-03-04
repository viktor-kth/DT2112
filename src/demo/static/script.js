/*
    The inspiration for this script comes from the following tutorial:
    https://www.youtube.com/watch?v=qs8FQ_oT33g
*/

const recordButton = window.document.getElementById('record');
const recordText = window.document.getElementById('record-text');
const output = window.document.getElementById('output');
const visualizer = document.getElementById('visualizer');
const ctx = visualizer.getContext('2d');

const celebrityList = document.getElementById('celebrityList');
let audioContext, analyser, source;

let freqs = [];

if (navigator.mediaDevices.getUserMedia) {
    console.log('getUserMedia supported.');

    let onMediaSetupSuccess = function(stream) {
        console.log('MediaRecorder started.');
        const mediaRecorder = new MediaRecorder(stream);
        let audioChunks = [];
        
        const audioContext = new AudioContext();
        const analyser = audioContext.createAnalyser();
        const source = audioContext.createMediaStreamSource(stream);
        source.connect(analyser);
        analyser.connect(audioContext.destination);

        
        freqs = new Uint8Array(analyser.frequencyBinCount);

        
        function draw() {
            // Clear the canvas
            
            ctx.clearRect(0, 0, visualizer.width, visualizer.height);
            
            let radius = 125;
            let bars = 120;
            
            ctx.beginPath();
            ctx.lineWidth = 3;
            ctx.arc(
                visualizer.width / 2,
                visualizer.height / 2,
                radius,
                0,
                Math.PI * 2
            );
            ctx.stroke();
            
            analyser.getByteFrequencyData(freqs);
            
            // Draw the bars

            if( mediaRecorder.state === 'recording') {
                for (let i = 0; i < bars; i++) {
                    let radians = (Math.PI * 2)  / bars;
                    let bar_height = freqs[i];
                    
                    let x_start = visualizer.width / 2 + Math.cos(radians * i) * (radius);
                    let y_start = visualizer.height / 2 + Math.sin(radians * i) * (radius);

                    let x_end = visualizer.width / 2 + Math.cos(radians * i) * (radius + bar_height);
                    let y_end = visualizer.height / 2 + Math.sin(radians * i) * (radius + bar_height);

                    let color = "rgb(" + 210 + ", " + (200 - freqs[i]) + ", " + freqs[i] + ")";
                    ctx.strokeStyle = color;
                    ctx.lineWidth = 5;
                    ctx.beginPath();
                    ctx.moveTo(x_start, y_start);
                    ctx.lineTo(x_end, y_end);
                    ctx.stroke();   
                    

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

function render_celebrity_list(celebrities) {
    
    /**
     * Render the celebrity list
     * @param {Array} celebrities - The list of celebrities
     * 
     * <div class="celebrity-item">
                    <img src="{{ url_for('static', filename='images/celebrity1.jpg') }}" alt="Celebrity 1" class="celebrity-image">
                    <h3 class="celebrity-name">Celebrity 1</h3>
                    <a href="#" class="wiki-link">View on Wikipedia</a>
                </div>
     */
    celebrityList.innerHTML = '';

    celebrities.forEach(celebrity => {
        const celebrityItem = document.createElement('div');
        celebrityItem.classList.add('celebrity-item');

        const celebrityImage = document.createElement('img');
        celebrityImage.src = `static/images/${celebrity.name}.jpg`;
        celebrityImage.alt = celebrity.name;
        celebrityImage.classList.add('celebrity-image');

        const celebrityName = document.createElement('h3');
        celebrityName.textContent = celebrity.name;
        celebrityName.classList.add('celebrity-name');

        const wikiLink = document.createElement('a');
        wikiLink.href = celebrity.wiki_url;
        wikiLink.textContent = 'View on Wikipedia';
        wikiLink.classList.add('wiki-link');

        celebrityItem.appendChild(celebrityImage);
        celebrityItem.appendChild(celebrityName);
        celebrityItem.appendChild(wikiLink);

        celebrityList.appendChild(celebrityItem);
    });
}
