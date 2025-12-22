// Animated background for HyprL Landing Page - Quant Lab (Pro Mode)
(function() {
    const canvas = document.getElementById('bg-canvas');
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    let width, height, dpr = 1;

    // Palette cohérente (même que track record portal)
    const colors = {
        accent1: '#7C3AED',  // violet
        accent2: '#22D3EE',  // cyan
        accent3: '#F472B6',  // pink
        accent4: '#34D399',  // green
        accent5: '#FBBF24',  // amber
    };

    // State management
    let isHovering = false;
    let scrollOffset = 0;
    let intensityMultiplier = 1.0;

    // Brownian motion state for scatter plot
    let brownianPrice = 100;
    let brownianVelocity = 0;
    const meanReversionSpeed = 0.02;
    const volatility = 0.8;

    // Mathematical functions (subtler)
    const functions = [
        {name: 'sin', fn: (x, t) => Math.sin(x * 0.015 + t * 0.0003) * 80, color: colors.accent1},
        {name: 'cos', fn: (x, t) => Math.cos(x * 0.012 + t * 0.00025) * 90, color: colors.accent2},
        {name: 'brownian', fn: (x, t) => {
            // OU process (mean-reverting Brownian)
            const mean = 0;
            const reversion = -meanReversionSpeed * brownianPrice;
            const noise = (Math.random() - 0.5) * volatility;
            return reversion + noise;
        }, color: colors.accent4},
    ];

    // Fibonacci levels
    const fibLevels = [0.236, 0.382, 0.5, 0.618, 0.786];

    // Floating equations (slower, collision-aware)
    const equations = [
        {text: 'E[R] = Σ(p·r)', x: 0, y: 0, vx: 0.15, vy: 0.1, color: colors.accent1, width: 120},
        {text: 'σ² = Var(R)', x: 0, y: 0, vx: -0.1, vy: 0.15, color: colors.accent2, width: 100},
        {text: 'S = (R-Rf)/σ', x: 0, y: 0, vx: 0.12, vy: -0.08, color: colors.accent3, width: 110},
        {text: 'PF = W/L', x: 0, y: 0, vx: -0.15, vy: -0.12, color: colors.accent4, width: 80},
        {text: 'f* = p-q/b', x: 0, y: 0, vx: 0.1, vy: 0.18, color: colors.accent5, width: 90},
        {text: 'MaxDD = Δpeak/peak', x: 0, y: 0, vx: 0.08, vy: -0.1, color: colors.accent1, width: 140},
        {text: 'CVaR = E[L|L≥VaR]', x: 0, y: 0, vx: -0.12, vy: 0.09, color: colors.accent2, width: 130},
    ];

    const MAX_VELOCITY = 0.2;

    // Initialize equations positions (with spacing to avoid overlap)
    function initEquations() {
        const margin = 100;
        equations.forEach((eq, i) => {
            eq.x = margin + (i * (width - 2 * margin) / equations.length);
            eq.y = margin + Math.random() * (height - 2 * margin);
        });
    }

    // B. Responsive + DPI correct
    function resize() {
        dpr = window.devicePixelRatio || 1;
        width = window.innerWidth;
        height = window.innerHeight;

        canvas.width = width * dpr;
        canvas.height = height * dpr;
        canvas.style.width = width + 'px';
        canvas.style.height = height + 'px';

        ctx.scale(dpr, dpr);
        initEquations();
    }

    // C. Focus states (hover/scroll)
    function setupInteractions() {
        const hero = document.querySelector('.hero');
        if (hero) {
            hero.addEventListener('mouseenter', () => {
                isHovering = true;
            });
            hero.addEventListener('mouseleave', () => {
                isHovering = false;
            });
        }

        window.addEventListener('scroll', () => {
            scrollOffset = window.scrollY;
            // Reduce intensity when scrolled down
            intensityMultiplier = Math.max(0.3, 1 - scrollOffset / 1000);
        });
    }

    // A. Draw grid (subtler, pro mode)
    function drawGrid() {
        const baseOpacity = 0.05 * intensityMultiplier;
        ctx.strokeStyle = `rgba(124, 58, 237, ${baseOpacity})`;
        ctx.lineWidth = 1;

        // Vertical lines (wider spacing)
        for (let x = 0; x < width; x += 150) {
            ctx.beginPath();
            ctx.moveTo(x, 0);
            ctx.lineTo(x, height);
            ctx.stroke();
        }

        // Horizontal lines
        for (let y = 0; y < height; y += 150) {
            ctx.beginPath();
            ctx.moveTo(0, y);
            ctx.lineTo(width, y);
            ctx.stroke();
        }
    }

    // F. Draw Fibonacci levels (very subtle)
    function drawFibonacci() {
        const baseOpacity = 0.08 * intensityMultiplier;
        ctx.lineWidth = 1;

        fibLevels.forEach((level, i) => {
            const y = height * level;
            ctx.strokeStyle = `rgba(34, 211, 238, ${baseOpacity})`;
            ctx.setLineDash([5, 10]);
            ctx.beginPath();
            ctx.moveTo(0, y);
            ctx.lineTo(width, y);
            ctx.stroke();
            ctx.setLineDash([]);
        });
    }

    // A. Draw mathematical functions (subtler opacity + glow)
    function drawFunctions(time) {
        const hoverBoost = isHovering ? 1.3 : 1.0;

        functions.forEach((func, index) => {
            ctx.beginPath();
            ctx.strokeStyle = func.color;
            ctx.lineWidth = 1.5;
            ctx.globalAlpha = (0.18 + 0.1 * hoverBoost) * intensityMultiplier;

            const yOffset = height / 2 + (index - functions.length / 2) * 100;

            for (let x = 0; x < width; x += 5) {
                const y = yOffset + func.fn(x, time);
                if (x === 0) {
                    ctx.moveTo(x, y);
                } else {
                    ctx.lineTo(x, y);
                }
            }

            ctx.stroke();

            // Reduced glow
            ctx.shadowColor = func.color;
            ctx.shadowBlur = 8 * intensityMultiplier;
            ctx.globalAlpha = 0.12 * intensityMultiplier;
            ctx.lineWidth = 3;
            ctx.stroke();

            ctx.shadowBlur = 0;
            ctx.globalAlpha = 1;
        });
    }

    // D. Draw floating equations (collisions + wrap + anti-overlap)
    function drawEquations(time) {
        const margin = 80;

        equations.forEach((eq, i) => {
            // Limit velocity
            eq.vx = Math.max(-MAX_VELOCITY, Math.min(MAX_VELOCITY, eq.vx));
            eq.vy = Math.max(-MAX_VELOCITY, Math.min(MAX_VELOCITY, eq.vy));

            // Update position
            eq.x += eq.vx * intensityMultiplier;
            eq.y += eq.vy * intensityMultiplier;

            // Wrap around (not just bounce)
            if (eq.x < -eq.width) eq.x = width;
            if (eq.x > width) eq.x = -eq.width;
            if (eq.y < -30) eq.y = height;
            if (eq.y > height) eq.y = -30;

            // Bounce off edges (for visible region)
            if (eq.x < margin || eq.x > width - eq.width - margin) eq.vx *= -0.9;
            if (eq.y < margin || eq.y > height - margin) eq.vy *= -0.9;

            // Simple repulsion from other equations (anti-overlap)
            equations.forEach((other, j) => {
                if (i !== j) {
                    const dx = eq.x - other.x;
                    const dy = eq.y - other.y;
                    const dist = Math.sqrt(dx * dx + dy * dy);
                    const minDist = 100;

                    if (dist < minDist && dist > 0) {
                        const force = (minDist - dist) * 0.01;
                        eq.vx += (dx / dist) * force;
                        eq.vy += (dy / dist) * force;
                    }
                }
            });

            // Draw equation
            ctx.save();
            ctx.font = '16px monospace';
            ctx.fillStyle = eq.color;
            const baseAlpha = 0.5 + Math.sin(time * 0.001 + i) * 0.15;
            ctx.globalAlpha = baseAlpha * intensityMultiplier;

            // Reduced glow
            ctx.shadowColor = eq.color;
            ctx.shadowBlur = 12 * intensityMultiplier;
            ctx.fillText(eq.text, eq.x, eq.y);

            ctx.restore();
        });
    }

    // E. Draw market-like scatter (random walk mean-reverting)
    let priceHistory = [];
    const maxHistory = 50;

    function drawDataPoints(time) {
        // Update Brownian motion (mean-reverting)
        const dt = 0.1;
        const mean = 100;
        const reversion = -meanReversionSpeed * (brownianPrice - mean);
        const noise = (Math.random() - 0.5) * volatility * Math.sqrt(dt);

        // Rare jump (news spike) - very subtle
        if (Math.random() < 0.002) {
            brownianPrice += (Math.random() - 0.5) * 10;
        }

        brownianPrice += reversion + noise;
        brownianPrice = Math.max(50, Math.min(150, brownianPrice));

        priceHistory.push(brownianPrice);
        if (priceHistory.length > maxHistory) priceHistory.shift();

        // Draw price path
        ctx.globalAlpha = 0.25 * intensityMultiplier;
        ctx.strokeStyle = colors.accent4;
        ctx.lineWidth = 2;
        ctx.beginPath();

        priceHistory.forEach((price, i) => {
            const x = (i / maxHistory) * width;
            const y = height * 0.7 - price;

            if (i === 0) {
                ctx.moveTo(x, y);
            } else {
                ctx.lineTo(x, y);
            }
        });

        ctx.stroke();

        // Draw points
        ctx.globalAlpha = 0.2 * intensityMultiplier;
        priceHistory.forEach((price, i) => {
            const x = (i / maxHistory) * width;
            const y = height * 0.7 - price;

            ctx.fillStyle = colors.accent4;
            ctx.beginPath();
            ctx.arc(x, y, 2, 0, Math.PI * 2);
            ctx.fill();
        });

        ctx.globalAlpha = 1;
        ctx.shadowBlur = 0;
    }

    // Draw gradient background (subtler)
    function drawGradientBg(time) {
        const gradient = ctx.createRadialGradient(
            width / 2, height / 2, 0,
            width / 2, height / 2, Math.max(width, height)
        );

        const offset = Math.sin(time * 0.0003) * 0.08;
        gradient.addColorStop(0, `rgba(11, 15, 25, 0.98)`);
        gradient.addColorStop(0.5 + offset, `rgba(124, 58, 237, ${0.03 * intensityMultiplier})`);
        gradient.addColorStop(1, `rgba(11, 15, 25, 1)`);

        ctx.fillStyle = gradient;
        ctx.fillRect(0, 0, width, height);
    }

    let lastTime = 0;
    const fps = 30;
    const frameDelay = 1000 / fps;

    function animate(currentTime) {
        if (currentTime - lastTime < frameDelay) {
            requestAnimationFrame(animate);
            return;
        }
        lastTime = currentTime;

        // Clear with gradient
        drawGradientBg(currentTime);

        // Draw grid
        drawGrid();

        // Draw Fibonacci levels
        drawFibonacci();

        // Draw functions
        drawFunctions(currentTime);

        // Draw market-like data
        drawDataPoints(currentTime);

        // Draw floating equations
        drawEquations(currentTime);

        requestAnimationFrame(animate);
    }

    // Initialize
    window.addEventListener('resize', resize);
    resize();
    setupInteractions();
    requestAnimationFrame(animate);
})();
