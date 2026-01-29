

const express = require("express");
const jwt = require("jsonwebtoken");
const bodyParser = require("body-parser");
const cors = require("cors");
const crypto = require("crypto");

const app = express();
app.use(bodyParser.json());
app.use(cors());

const SECRET_KEY = "gt77tgygiugy8795f878";
const PORT = 5000;

// ------------------------
// In-memory DB
// ------------------------
let usersDB = {};
let modelsDB = {};
let trainingQueue = [];

// ------------------------
// Schemas (for clarity)
// ------------------------
class User {
    constructor(username, email, password, role="user"){
        this.username = username;
        this.email = email;
        this.password = password;
        this.role = role;
    }
}

class MetricsResponse {
    constructor(epoch, loss, accuracy, precision, recall, f1){
        this.epoch = epoch;
        this.loss = loss;
        this.accuracy = accuracy;
        this.precision = precision;
        this.recall = recall;
        this.f1 = f1;
    }
}

// ------------------------
// AUTH FUNCTIONS
// ------------------------
function hashPassword(password){
    return crypto.createHash("sha256").update(password).digest("hex");
}

function verifyPassword(password, hash){
    return hashPassword(password) === hash;
}

function generateToken(username){
    return jwt.sign({username}, SECRET_KEY, {expiresIn: "1h"});
}

function authenticateToken(req, res, next){
    const authHeader = req.headers["authorization"];
    if(!authHeader) return res.sendStatus(401);
    const token = authHeader.split(" ")[1];
    jwt.verify(token, SECRET_KEY, (err, user)=>{
        if(err) return res.sendStatus(403);
        req.user = user;
        next();
    });
}

// ------------------------
// AUTH ENDPOINTS
// ------------------------
app.post("/signup", (req,res)=>{
    const {username,email,password,role} = req.body;
    if(usersDB[username]) return res.status(400).send("Username exists");
    usersDB[username] = new User(username,email,hashPassword(password),role || "user");
    res.send("User created successfully");
});

app.post("/login", (req,res)=>{
    const {username,password} = req.body;
    const user = usersDB[username];
    if(!user || !verifyPassword(password,user.password)) return res.status(401).send("Invalid credentials");
    const token = generateToken(username);
    res.json({token});
});

// ------------------------
// TRAINING SIMULATION
// ------------------------
function simulateDataset(samples=1000, features=10, classes=2){
    let X = [];
    let y = [];
    for(let i=0;i<samples;i++){
        let row = [];
        for(let f=0;f<features;f++){
            row.push(Math.random());
        }
        X.push(row);
        y.push(Math.floor(Math.random()*classes));
    }
    return {X,y};
}

function calculateMetrics(yTrue, yPred){
    let total = yTrue.length;
    let correct = yTrue.filter((v,i)=>v===yPred[i]).length;
    let accuracy = correct/total;
    let precision = Math.random(); // simulate
    let recall = Math.random();
    let f1 = Math.random();
    return {accuracy, precision, recall, f1};
}

async function trainModel(modelName, modelType="mlp", epochs=5, batchSize=32){
    const {X,y} = simulateDataset(1000, 10, 2);
    let metrics = [];
    for(let e=1;e<=epochs;e++){
        // Simulate training delay
        await new Promise(r=>setTimeout(r,50));
        let yPred = y.map(v=>Math.round(Math.random())); // simulate prediction
        let m = calculateMetrics(y,yPred);
        metrics.push(new MetricsResponse(e, Math.random(), m.accuracy, m.precision, m.recall, m.f1));
    }
    let version = crypto.randomBytes(3).toString("hex");
    modelsDB[modelName] = {type:modelType, metrics, version, model:{}}; // model:{} placeholder
    console.log(`Model ${modelName} trained, version ${version}`);
}

// ------------------------
// TRAINING ENDPOINT
// ------------------------
app.post("/train", authenticateToken, async (req,res)=>{
    const {modelName, modelType, epochs, batchSize} = req.body;
    const jobId = crypto.randomBytes(4).toString("hex");
    trainingQueue.push({jobId, modelName, status:"queued", user:req.user.username});
    trainModel(modelName, modelType || "mlp", epochs || 5, batchSize || 32).then(()=>{
        const job = trainingQueue.find(j=>j.jobId===jobId);
        if(job) job.status="completed";
    });
    res.json({jobId, status:"queued", modelName});
});

// ------------------------
// PREDICTION ENDPOINT
// ------------------------
app.post("/predict", authenticateToken, (req,res)=>{
    const {modelName, features} = req.body;
    const model = modelsDB[modelName];
    if(!model) return res.status(400).send("Model not found");
    let prediction = Math.floor(Math.random()*2); // simulate
    res.json({prediction});
});

// ------------------------
// METRICS ENDPOINT
// ------------------------
app.get("/metrics/:modelName", authenticateToken, (req,res)=>{
    const model = modelsDB[req.params.modelName];
    if(!model) return res.status(404).send("Model not found");
    res.json(model.metrics);
});

// ------------------------
// MASSIVE TRAINING SIMULATION
// ------------------------
app.get("/simulate_massive_training", authenticateToken, (req,res)=>{
    for(let i=0;i<5;i++){
        let modelName = `sim_model_${i}_${Date.now()}`;
        trainModel(modelName,"mlp",10,64);
    }
    res.send("Massive training started");
});

// ------------------------
app.listen(PORT,()=>console.log(`MIT-level AI/ML backend running on port ${PORT}`));
