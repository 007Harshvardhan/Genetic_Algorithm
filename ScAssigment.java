import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.concurrent.ThreadLocalRandom;
import java.util.stream.IntStream;

public class ScAssigment {
    static final int POP_SIZE = 100; // Population size
    static final int N = 50;         // Length of vector x (genes per chromosome)
    static final int M = 20;         // Length of vector y
    static final int MAX_GEN = 1000; // Maximum number of generations
    static final double CROSSOVER_RATE = 0.7; // Crossover probability
    static final double MUTATION_RATE = 0.01; // Mutation probability
    static final int K = 5; // Number of non-zero entries in x

    // Random number generators
    static Random random = new Random();
    static ThreadLocalRandom tlr = ThreadLocalRandom.current();

    // Function to generate a random sparse vector x with Gaussian non-zero values
    static double[] generateSparseVector(int N, int K) {
        double[] x = new double[N];
        Arrays.fill(x, 0.0);

        for (int i = 0; i < K; i++) {
            int index = random.nextInt(N);
            x[index] = tlr.nextGaussian() * 1 + 0.05; // Random non-zero value from N(0.05, 1)
        }
        return x;
    }

    // Function to calculate the cost (fitness) J(x) = ||y - Hx||_2^2
    static double costFunction(double[] y, double[][] H, double[] x) {
        double error = 0.0;
        double[] residual = new double[y.length];

        for (int i = 0; i < y.length; i++) {
            residual[i] = y[i];
            for (int j = 0; j < x.length; j++) {
                residual[i] -= H[i][j] * x[j];
            }
            error += residual[i] * residual[i];
        }

        return error; // Reconstruction error ||y - Hx||_2^2
    }

    // Selection operator (tournament selection)
    static double[] tournamentSelection(List<double[]> population, double[] fitness, double[] y, double[][] H) {
        int tournamentSize = 3;
        double[] best = population.get(random.nextInt(POP_SIZE));

        for (int i = 1; i < tournamentSize; i++) {
            int randomIndex = random.nextInt(POP_SIZE);
            if (fitness[randomIndex] < costFunction(y, H, population.get(randomIndex))) {
                best = population.get(randomIndex);
            }
        }
        return best;
    }

    // Crossover operator (single-point crossover)
    static void crossover(double[] parent1, double[] parent2) {
        if (tlr.nextDouble() < CROSSOVER_RATE) {
            int point = random.nextInt(N);
            for (int i = point; i < N; i++) {
                double temp = parent1[i];
                parent1[i] = parent2[i];
                parent2[i] = temp;
            }
        }
    }

    // Mutation operator
    static void mutate(double[] chromosome) {
        for (int i = 0; i < N; i++) {
            if (tlr.nextDouble() < MUTATION_RATE) {
                chromosome[i] += tlr.nextDouble() - 0.5;
            }
        }
    }

    // Genetic Algorithm
    static double[] geneticAlgorithm(double[] y, double[][] H) {
        List<double[]> population = new ArrayList<>(POP_SIZE);
        double[] fitness = new double[POP_SIZE];

        // Step 1: Initialize population
        for (int i = 0; i < POP_SIZE; i++) {
            double[] individual = generateSparseVector(N, K);
            population.add(individual);
            fitness[i] = costFunction(y, H, individual);
        }

        // Step 2: Evolution loop
        for (int gen = 0; gen < MAX_GEN; gen++) {
            List<double[]> newPopulation = new ArrayList<>(POP_SIZE);

            // Step 3: Selection, Crossover, and Mutation
            for (int i = 0; i < POP_SIZE; i += 2) {
                double[] parent1 = tournamentSelection(population, fitness, y, H);
                double[] parent2 = tournamentSelection(population, fitness, y, H);
                crossover(parent1, parent2);
                mutate(parent1);
                mutate(parent2);
                newPopulation.add(parent1);
                newPopulation.add(parent2);
            }

            // Step 4: Evaluate new population
            for (int i = 0; i < POP_SIZE; i++) {
                fitness[i] = costFunction(y, H, newPopulation.get(i));
            }

            population = newPopulation;

            // Step 5: Check for convergence
            if (gen % 100 == 0) {
                double bestCost = Arrays.stream(fitness).min().orElse(Double.MAX_VALUE);
                System.out.println("Generation " + gen + ", Best Cost: " + bestCost);
            }
        }

        // Step 6: Return the best solution found
        int bestIndex = IntStream.range(0, POP_SIZE).reduce((i, j) -> fitness[i] < fitness[j] ? i : j).orElse(0);
        return population.get(bestIndex);
    }

    public static void main(String[] args) {
        double[][] H = new double[M][N];
        double[] y = new double[M];

        // Generate random Gaussian dictionary matrix H
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                H[i][j] = tlr.nextGaussian(); // H follows N(0, 1)
            }
        }

        // Generate random measurement vector y
        for (int i = 0; i < M; i++) {
            y[i] = tlr.nextDouble(); // Example random measurement, can replace with real data
        }

        // Apply Genetic Algorithm to find sparse x
        double[] bestSolution = geneticAlgorithm(y, H);

        // Output the best solution
        System.out.println("Best Solution (Sparse x):");
        for (double val : bestSolution) {
            System.out.print(val + " ");
        }
        System.out.println();
    }
}
