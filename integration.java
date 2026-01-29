// MainApplication.java
package com.mitlevelbackend;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.web.bind.annotation.*;
import org.springframework.http.ResponseEntity;
import org.springframework.scheduling.annotation.Async;
import org.springframework.scheduling.annotation.EnableAsync;
import org.springframework.context.annotation.Bean;
import org.springframework.scheduling.concurrent.ThreadPoolTaskExecutor;

import java.util.*;
import java.util.concurrent.*;
import java.util.stream.Collectors;

@SpringBootApplication
@EnableAsync
@RestController
@RequestMapping("/api")
public class MainApplication {

    // ----------------------------
    // GLOBALS / IN-MEMORY DB
    // ----------------------------
    private Map<String, User> userDB = new ConcurrentHashMap<>();
    private Map<String, ModelEntry> modelDB = new ConcurrentHashMap<>();
    private Queue<TrainingJob> trainingQueue = new ConcurrentLinkedQueue<>();

    public static void main(String[] args) {
        SpringApplication.run(MainApplication.class, args);
    }

    // ----------------------------
    // ASYNC EXECUTOR
    // ----------------------------
    @Bean
    public Executor taskExecutor() {
        ThreadPoolTaskExecutor executor = new ThreadPoolTaskExecutor();
        executor.setCorePoolSize(6);
        executor.setMaxPoolSize(12);
        executor.setQueueCapacity(50);
        executor.initialize();
        return executor;
    }

    // ----------------------------
    // USER CLASSES
    // ----------------------------
    static class User {
        String username;
        String email;
        String password;
        String role;

        public User(String u, String e, String p, String r) {
            username = u; email = e; password = p; role = r;
        }
    }

    static class AuthRequest {
        public String username;
        public String password;
        public String email;
        public String role;
    }

    static class AuthResponse {
        public String token;
        public AuthResponse(String t) { token = t; }
    }

    // ----------------------------
    // MODEL / TRAINING CLASSES
    // ----------------------------
    static class TrainRequest {
        public String modelName;
        public int epochs;
        public double lr;
        public int batchSize;
        public String modelType; // mlp, cnn, dt
    }

    static class PredictRequest {
        public String modelName;
        public List<Double> features;
    }

    static class MetricsResponse {
        public int epoch;
        public double loss;
        public double accuracy;
        public double precision;
        public double recall;
        public double f1Score;

        public MetricsResponse(int e, double l, double a, double p, double r, double f) {
            epoch = e; loss = l; accuracy = a; precision = p; recall = r; f1Score = f;
        }
    }

    static class ModelEntry {
        public String type;
        public List<MetricsResponse> metrics = new ArrayList<>();
        public String version;
        public Object model;
    }

    static class TrainingJob {
        public String jobId;
        public String modelName;
        public String status;
        public String user;

        public TrainingJob(String jid, String mn, String s, String u){
            jobId = jid; modelName = mn; status = s; user = u;
        }
    }

    // ----------------------------
    // AUTH ENDPOINTS
    // ----------------------------

    @PostMapping("/signup")
    public ResponseEntity<String> signup(@RequestBody AuthRequest req) {
        if(userDB.containsKey(req.username))
            return ResponseEntity.badRequest().body("Username exists");
        userDB.put(req.username, new User(req.username, req.email, req.password, req.role));
        return ResponseEntity.ok("User created");
    }

    @PostMapping("/login")
    public ResponseEntity<AuthResponse> login(@RequestBody AuthRequest req) {
        User user = userDB.get(req.username);
        if(user == null || !user.password.equals(req.password))
            return ResponseEntity.status(401).build();
        String token = Base64.getEncoder().encodeToString((req.username+":token").getBytes());
        return ResponseEntity.ok(new AuthResponse(token));
    }

    // ----------------------------
    // TRAINING ENDPOINTS
    // ----------------------------

    @PostMapping("/train")
    public ResponseEntity<String> train(@RequestBody TrainRequest req, @RequestHeader("Authorization") String auth){
        String jobId = UUID.randomUUID().toString();
        trainingQueue.add(new TrainingJob(jobId, req.modelName, "queued", auth));
        asyncTrain(req, auth, jobId);
        return ResponseEntity.ok("Training started with JobID: "+jobId);
    }

    @Async
    public void asyncTrain(TrainRequest req, String userToken, String jobId) {
        try{
            Thread.sleep(1000); // simulate setup
            ModelEntry entry = new ModelEntry();
            entry.type = req.modelType;
            entry.version = UUID.randomUUID().toString();
            List<MetricsResponse> metrics = new ArrayList<>();
            Random rnd = new Random();

            for(int epoch=1; epoch<=req.epochs; epoch++){
                double loss = rnd.nextDouble();
                double acc = rnd.nextDouble();
                double precision = rnd.nextDouble();
                double recall = rnd.nextDouble();
                double f1 = rnd.nextDouble();
                metrics.add(new MetricsResponse(epoch, loss, acc, precision, recall, f1));
                Thread.sleep(100); // simulate batch processing
            }

            entry.metrics = metrics;
            entry.model = new Object(); // placeholder for real ML model
            modelDB.put(req.modelName, entry);
            // Update queue status
            for(TrainingJob job : trainingQueue)
                if(job.jobId.equals(jobId)) job.status = "completed";

        }catch(Exception e){ e.printStackTrace(); }
    }

    @PostMapping("/predict")
    public ResponseEntity<Integer> predict(@RequestBody PredictRequest req){
        ModelEntry entry = modelDB.get(req.modelName);
        if(entry == null) return ResponseEntity.badRequest().body(-1);
        Random rnd = new Random();
        return ResponseEntity.ok(rnd.nextInt(3)); // simulated prediction
    }

    @GetMapping("/metrics/{modelName}")
    public ResponseEntity<List<MetricsResponse>> getMetrics(@PathVariable String modelName){
        ModelEntry entry = modelDB.get(modelName);
        if(entry == null) return ResponseEntity.notFound().build();
        return ResponseEntity.ok(entry.metrics);
    }

    @GetMapping("/simulate_massive_training")
    public ResponseEntity<String> simulateMassive(@RequestHeader("Authorization") String auth){
        for(int i=0;i<5;i++){
            TrainRequest req = new TrainRequest();
            req.modelName = "sim_model_"+i+"_"+System.currentTimeMillis();
            req.epochs = 10; req.lr = 0.001; req.batchSize = 64; req.modelType="mlp";
            String jobId = UUID.randomUUID().toString();
            trainingQueue.add(new TrainingJob(jobId, req.modelName, "queued", auth));
            asyncTrain(req, auth, jobId);
        }
        return ResponseEntity.ok("Massive training started");
    }
}
