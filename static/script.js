const video = document.getElementById("camera");
const canvas = document.getElementById("canvas");
const statusText = document.getElementById("status");
const cameraContainer = document.getElementById("camera-container");

// Start Camera Stream
async function startCamera() {
    const cameraContainer = document.getElementById("camera-container");
    const video = document.getElementById("camera");

    if (!cameraContainer || !video) {
        console.error("Camera elements not found!");
        return;
    }

    try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        video.srcObject = stream;
        cameraContainer.style.display = "block"; // Show the camera
    } catch (error) {
        console.error("Error accessing camera:", error);
        alert("Camera access denied. Please enable camera permissions.");
    }
}





// Capture Image from Video Feed
async function captureImage() {
    const video = document.getElementById("camera");
    const canvas = document.getElementById("canvas");
    const ctx = canvas.getContext("2d");

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    const blob = await new Promise(resolve => canvas.toBlob(resolve, "image/jpeg"));

    console.log("📸 Captured Image:", blob); // 🔍 Check if image is captured
    return blob;
}


// Start Camera on Page Load
startCamera();

document.getElementById("register-btn").addEventListener("click", async () => {
    const student_id = prompt("Enter Student ID:");
    const name = prompt("Enter Name:");
    if (!student_id || !name) return alert("Invalid input");

    await startCamera(); // Ensure the camera is running before capturing

    let images = [];
    for (let i = 0; i < 11; i++) {
        const image = await captureImage();
        images.push(image);
        console.log(`Captured image ${i + 1}`);
        await new Promise(resolve => setTimeout(resolve, 500)); // Delay between captures
    }

    const formData = new FormData();
    formData.append("student_id", student_id);
    formData.append("name", name);
    images.forEach((img, index) => formData.append(`image${index}`, img));

    fetch("/register", { method: "POST", body: formData })
    .then(res => res.json())
    .then(data => {
        console.log(data); // Debugging
        document.getElementById("status").innerText = data.message || "Registration complete!";
    })
    .catch(error => {
        console.error("Error:", error);
        document.getElementById("status").innerText = "Registration failed.";
    });

});




// Verify Face
document.getElementById("verify-btn").addEventListener("click", async () => {
    const image = await captureImage();
    const formData = new FormData();
    formData.append("image", image);

    console.log("📸 Sending image for verification...");

    fetch("/verify", { method: "POST", body: formData })
        .then(res => res.json())
        .then(data => {
            console.log("🔍 Verification Response:", data);
            if (data.verified) {
                statusText.innerText = `✅ Verified: ${data.name}`;
            } else {
                statusText.innerText = "❌ Not Recognized";
            }
        })
        .catch(err => console.error("❌ Verification Error:", err));
});


document.getElementById("attendance-btn").addEventListener("click", async () => {
    const image = await captureImage();
    const formData = new FormData();
    formData.append("image", image);

    console.log("📸 Sending image for attendance...");

    fetch("/mark-attendance", { method: "POST", body: formData })
        .then(res => res.json())
        .then(data => {
            console.log("📊 Attendance Response:", data);
            statusText.innerText = data.message;
        })
        .catch(err => console.error("❌ Attendance Error:", err));
});

