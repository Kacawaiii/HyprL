const canvas = document.getElementById('bg-canvas');
const ctx = canvas.getContext('2d');

let width, height;
let clouds = [];
let time = 0;

const palette = {
    skyTop: '#b8e9ff',    // sky-cyan
    skyMid: '#d8b4fe',    // sky-lavender
    skyLow: '#fbcfe8',    // sky-rose
    skyBottom: '#fef3c7'  // sky-gold
};

function resize() {
    width = canvas.width = window.innerWidth;
    height = canvas.height = window.innerHeight;
    initClouds();
}

function initClouds() {
    clouds = [];
    const count = 8;
    for (let i = 0; i < count; i++) {
        const radius = 100 + Math.random() * 150;
        const lobesCount = 5 + Math.floor(Math.random() * 5);
        const lobes = [];
        for (let j = 0; j < lobesCount; j++) {
            lobes.push({
                x: (Math.random() - 0.5) * 1.5,
                y: (Math.random() - 0.5) * 0.8
            });
        }
        clouds.push({
            x: Math.random() * width,
            y: Math.random() * height * 0.6,
            radius: radius,
            speed: 0.2 + Math.random() * 0.5,
            seed: Math.random() * 1000,
            bob: 20 + Math.random() * 40,
            lobes: lobes
        });
    }
}

function drawBackground() {
    const gradient = ctx.createLinearGradient(0, 0, 0, height);
    gradient.addColorStop(0, palette.skyTop);
    gradient.addColorStop(0.42, palette.skyMid);
    gradient.addColorStop(0.7, palette.skyLow);
    gradient.addColorStop(1, palette.skyBottom);
    ctx.fillStyle = gradient;
    ctx.fillRect(0, 0, width, height);
}

function drawClouds(time, step) {
    ctx.save();
    ctx.globalCompositeOperation = 'screen';
    // On réduit légèrement le flou pour la performance
    ctx.filter = 'blur(30px)';
    
    clouds.forEach((cloud) => {
        cloud.lobes.forEach((lobe) => {
            const r = cloud.radius;
            const x = cloud.x + lobe.x * r;
            const y = (cloud.y + Math.sin(time * 0.001 + cloud.seed) * cloud.bob) + lobe.y * r;
            
            const puff = ctx.createRadialGradient(x, y, r * 0.15, x, y, r);
            puff.addColorStop(0, 'rgba(255, 255, 255, 0.8)');
            puff.addColorStop(0.6, 'rgba(255, 255, 255, 0.3)');
            puff.addColorStop(1, 'rgba(255, 255, 255, 0)');
            
            ctx.fillStyle = puff;
            ctx.beginPath();
            ctx.arc(x, y, r, 0, Math.PI * 2);
            ctx.fill();
        });
        
        // Mouvement horizontal
        cloud.x += cloud.speed;
        if (cloud.x - cloud.radius * 2 > width) {
            cloud.x = -cloud.radius * 2;
        }
    });
    ctx.restore();
}

function animate() {
    drawBackground();
    drawClouds(time, 1);
    time += 1;
    requestAnimationFrame(animate);
}

window.addEventListener('resize', resize);
resize();
animate();