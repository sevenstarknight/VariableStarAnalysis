/*
 * Copyright (C) 2016 Kyle Johnston
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
package fit.astro.vsa.common.utilities.swarm;

import org.apache.commons.math3.random.MersenneTwister;
import org.apache.commons.math3.random.RandomGenerator;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.cpu.nativecpu.NDArray;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Represents a swarm of particles from the Particle ParticleSwarmOptimization
 * Optimization algorithm.
 * <p>
 * Kennedy, J. (2011). Particle swarm optimization. In Encyclopedia of machine
 * learning (pp. 760-766). Springer, Boston, MA.
 *
 * @author LazoCoder, from https://github.com/LazoCoder
 * @author kjohnston, Updates, clean up, adjustments to Java8
 */
public class ParticleSwarmOptimization {

    // ======================================================
    // Inputs
    private final SwarmOptimizationFunction function; // The function to search.

    private final int dimensions, numOfParticles, epochs;

    // ======================================================
    // Adjustable
    /**
     * Parameters that define operations for the particles
     */
    private double inertialComponent = 0.729844;
    private double cognitiveComponent = 1.496180;
    private double socialComponent = 1.496180;

    /**
     * The Random Generator to dictate operations
     */
    private RandomGenerator rand = new MersenneTwister();

    /**
     * When Particles are created they are given a random position. The random
     * position is selected from a specified range. If the lower range is 0 and
     * the upper range is 10 then the value will be between 0 (inclusive) and 10
     * (exclusive).
     */
    private double upperRange = 101.0;
    private double lowerRange = -100.0;

    // ======================================================
    // Internal
    private INDArray bestPosition;
    private double bestEval;

    private static final Logger LOGGER
            = LoggerFactory.getLogger(ParticleSwarmOptimization.class);

    /**
     * Construct the Swarm.
     *
     * @param function The optimization function the swarm is trying to resolve
     * @param particles the number of particles to create
     * @param epochs the number of generations
     */
    public ParticleSwarmOptimization(SwarmOptimizationFunction function, int dimensions,
            int particles, int epochs) {
        this.numOfParticles = particles;
        this.dimensions = dimensions;
        this.epochs = epochs;
        this.function = function;

        bestPosition = new NDArray();
        bestEval = Double.POSITIVE_INFINITY;
    }

    /**
     * Execute the algorithm. Generate estimate of global optimum within the
     * defined bounds
     *
     * @return The "best" position based on search
     */
    public INDArray run() {
        Particle[] particles = initialize();

        double oldEval = bestEval;

        LOGGER.info("--------------------------EXECUTING-------------------------");
        LOGGER.info("Global Best Evaluation (Epoch " + 0 + "):\t" + bestEval);

        for (int i = 0; i < epochs; i++) {

            if (bestEval < oldEval) {

                LOGGER.info("Global Best Evaluation (Epoch " + (i + 1) + "):\t" + bestEval);
                oldEval = bestEval;
            }

            for (Particle p : particles) {
                p.updatePersonalBest();
                updateGlobalBest(p);
            }

            for (Particle p : particles) {
                updateVelocity(p);
                p.updatePosition();
            }
        }

        LOGGER.info("---------------------------RESULT---------------------------");
        LOGGER.info("Position = " + bestPosition.toString());
        LOGGER.info("Final Best Evaluation: " + bestEval);
        LOGGER.info("---------------------------COMPLETE-------------------------");

        return bestPosition;
    }

    /**
     * Create a set of particles, each with random starting positions.
     *
     * @return an array of particles
     */
    private Particle[] initialize() {
        Particle[] particles = new Particle[numOfParticles];
        for (int i = 0; i < numOfParticles; i++) {
            Particle particle = new Particle(function, dimensions);
            particle.setUpperLower(upperRange, lowerRange);
            particle.setRand(rand);

            particles[i] = particle;
            updateGlobalBest(particle);
        }
        return particles;
    }

    /**
     * Update the global best solution if a the specified particle has a better
     * solution
     *
     * @param particle the particle to analyze
     */
    private void updateGlobalBest(Particle particle) {
        if (particle.getBestEval() < bestEval) {
            bestPosition = particle.getBestPosition();
            bestEval = particle.getBestEval();
        }
    }

    /**
     * Update the velocity of a particle using the velocity update formula
     *
     * @param particle the particle to update
     */
    private void updateVelocity(Particle particle) {

        INDArray pBest = particle.getBestPosition();
        INDArray gBest = bestPosition.dup();

        // The first product of the formula.
        INDArray newVelocity = particle.getVelocity().dup();
        newVelocity.mul(inertialComponent);

        // The second product of the formula.
        pBest.sub(particle.getPosition());
        pBest.mul(cognitiveComponent);
        pBest.mul(rand.nextDouble());
        newVelocity.add(pBest);

        // The third product of the formula.
        gBest.sub(particle.getPosition());
        gBest.mul(socialComponent);
        gBest.mul(rand.nextDouble());
        newVelocity.add(gBest);

        particle.setVelocity(newVelocity);
    }

    // ==============================================================
    // Adjustable
    /**
     *
     * @param inertialComponent the particles resistance to change
     */
    public void setInertialComponent(double inertialComponent) {
        this.inertialComponent = inertialComponent;
    }

    /**
     *
     * @param cognitiveComponent the cognitive component or introversion of the
     * particle
     */
    public void setCognitiveComponent(double cognitiveComponent) {
        this.cognitiveComponent = cognitiveComponent;
    }

    /**
     *
     * @param socialComponent the social component or extroversion of the
     * particle
     */
    public void setSocialComponent(double socialComponent) {
        this.socialComponent = socialComponent;
    }

    public double getInertialComponent() {
        return inertialComponent;
    }

    public double getCognitiveComponent() {
        return cognitiveComponent;
    }

    public double getSocialComponent() {
        return socialComponent;
    }

    // =================
    public void setUpperRange(double upperRange) {
        this.upperRange = upperRange;
    }

    public void setLowerRange(double lowerRange) {
        this.lowerRange = lowerRange;
    }

    // =================
    public void setRand(RandomGenerator rand) {
        this.rand = rand;
    }

}
