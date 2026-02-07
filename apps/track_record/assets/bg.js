(function() {
    // Create Canvas
    const canvas = document.createElement('canvas');
    canvas.id = 'bg-canvas';
    canvas.style.position = 'fixed';
    canvas.style.top = '0';
    canvas.style.left = '0';
    canvas.style.width = '100vw';
    canvas.style.height = '100vh';
    canvas.style.zIndex = '-1';
    canvas.style.pointerEvents = 'none';
    document.body.appendChild(canvas);

    const ctx = canvas.getContext('2d');
    let width, height;
    let particles = [];
    
    // Check for reduced motion
    const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

    // Config
    const PARTICLE_COUNT = reducedMotion ? 20 : 60;
    const CONNECT_DISTANCE = 150;
    const MOUSE_RADIUS = 200;

    let mouse = { x: null, y: null };

    window.addEventListener('mousemove', (e) => {
        mouse.x = e.x;
        mouse.y = e.y;
    });

    function resize() {
        width = window.innerWidth;
        height = window.innerHeight;
        canvas.width = width;
        canvas.height = height;
        initParticles();
    }

    class Particle {
        constructor() {
            this.x = Math.random() * width;
            this.y = Math.random() * height;
            this.vx = (Math.random() - 0.5) * 0.5;
            this.vy = (Math.random() - 0.5) * 0.5;
            this.size = Math.random() * 2 + 1;
            // Colors: Cyan, Violet, Pinkish
            const colors = ['rgba(34, 211, 238, ', 'rgba(124, 58, 237, ', 'rgba(244, 114, 182, '];
            this.colorBase = colors[Math.floor(Math.random() * colors.length)];
            this.alpha = Math.random() * 0.5 + 0.1;
        }

        update() {
            if (reducedMotion) return;

            this.x += this.vx;
            this.y += this.vy;

            // Bounce off edges
            if (this.x < 0 || this.x > width) this.vx *= -1;
            if (this.y < 0 || this.y > height) this.vy *= -1;

            // Mouse interaction
            if (mouse.x != null) {
                let dx = mouse.x - this.x;
                let dy = mouse.y - this.y;
                let distance = Math.sqrt(dx * dx + dy * dy);
                if (distance < MOUSE_RADIUS) {
                    const forceDirectionX = dx / distance;
                    const forceDirectionY = dy / distance;
                    const force = (MOUSE_RADIUS - distance) / MOUSE_RADIUS;
                    const directionX = forceDirectionX * force * this.size;
                    const directionY = forceDirectionY * force * this.size;
                    this.x -= directionX;
                    this.y -= directionY;
                }
            }
        }

        draw() {
            ctx.beginPath();
            ctx.arc(this.x, this.y, this.size, 0, Math.PI * 2);
            ctx.fillStyle = this.colorBase + this.alpha + ')';
            ctx.fill();
        }
    }

    function initParticles() {
        particles = [];
        for (let i = 0; i < PARTICLE_COUNT; i++) {
            particles.push(new Particle());
        }
    }

    function animate() {
        ctx.clearRect(0, 0, width, height);
        
        // Draw Aurora Gradient Background
        const time = Date.now() * 0.0002;
        const gradient = ctx.createLinearGradient(0, 0, width, height);
        gradient.addColorStop(0, '#0b0f19');
        gradient.addColorStop(0.5, '#0f172a'); // Slate 900
        gradient.addColorStop(1, '#0b0f19');
        ctx.fillStyle = gradient;
        ctx.fillRect(0, 0, width, height);

        // Add subtle blobs
        if (!reducedMotion) {
            ctx.globalCompositeOperation = 'lighter';
            
            // Blob 1 (Cyan/Blue)
            const x1 = Math.sin(time) * width * 0.3 + width * 0.5;
            const y1 = Math.cos(time * 0.8) * height * 0.3 + height * 0.5;
            const rad1 = 400;
            const g1 = ctx.createRadialGradient(x1, y1, 0, x1, y1, rad1);
            g1.addColorStop(0, 'rgba(34, 211, 238, 0.08)');
            g1.addColorStop(1, 'rgba(0,0,0,0)');
            ctx.fillStyle = g1;
            ctx.fillRect(0, 0, width, height);

            // Blob 2 (Violet)
            const x2 = Math.cos(time * 1.1) * width * 0.3 + width * 0.5;
            const y2 = Math.sin(time * 0.9) * height * 0.3 + height * 0.5;
            const rad2 = 350;
            const g2 = ctx.createRadialGradient(x2, y2, 0, x2, y2, rad2);
            g2.addColorStop(0, 'rgba(124, 58, 237, 0.08)');
            g2.addColorStop(1, 'rgba(0,0,0,0)');
            ctx.fillStyle = g2;
            ctx.fillRect(0, 0, width, height);
            
            ctx.globalCompositeOperation = 'source-over';
        }

        // Draw Particles & Connections
        particles.forEach(p => {
            p.update();
            p.draw();
        });

        // Draw connections
        ctx.strokeStyle = 'rgba(148, 163, 184, 0.05)';
        for (let a = 0; a < particles.length; a++) {
            for (let b = a; b < particles.length; b++) {
                let dx = particles[a].x - particles[b].x;
                let dy = particles[a].y - particles[b].y;
                let distance = Math.sqrt(dx * dx + dy * dy);

                if (distance < CONNECT_DISTANCE) {
                    ctx.lineWidth = 1 - (distance / CONNECT_DISTANCE);
                    ctx.beginPath();
                    ctx.moveTo(particles[a].x, particles[a].y);
                    ctx.lineTo(particles[b].x, particles[b].y);
                    ctx.stroke();
                }
            }
        }

        requestAnimationFrame(animate);
    }

    window.addEventListener('resize', resize);
    resize();
    animate();
})();
