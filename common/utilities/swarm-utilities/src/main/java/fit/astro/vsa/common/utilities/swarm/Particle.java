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

import org.apache.commons.math3.distribution.UniformRealDistribution;
import org.apache.commons.math3.random.MersenneTwister;
import org.apache.commons.math3.random.RandomGenerator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.cpu.nativecpu.NDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Represents a particle from the Particle Swarm Optimization algorithm.
 *
 * @author LazoCoder, from https://github.com/LazoCoder
 * @author kjohnston, Updates, clean up, adjustments to Java8
 */
public class Particle {

    // ================================================
    // Input
    private final SwarmOptimizationFunction function;  // The evaluation function to use.
    private final int dimensions;
    // ===============================================
    // Internal
    private INDArray position;        // Current position.
    private INDArray velocity;
    private INDArray bestPosition;    // Personal best solution.

    private double bestEval;        // Personal best value.

    // ================================================
    // Adjustable
    private RandomGenerator rand = new MersenneTwister();

    private double upperRange = Double.MAX_VALUE;
    private double lowerRange = Double.MIN_VALUE;

    /**
     * Construct a Particle with a random starting position.
     *
     * @param function Function to be optimized
     * @param dimensions Coordinate R^n
     */
    public Particle(SwarmOptimizationFunction function, int dimensions) {

        this.function = function;
        this.dimensions = dimensions;

        // ===================================
        position = new NDArray();
        velocity = new NDArray();
        setRandomPosition();
        bestPosition = velocity.dup();
        bestEval = function.evaluate(position);
    }

    private void setRandomPosition() {

        UniformRealDistribution distribution
                = new UniformRealDistribution(rand, lowerRange, upperRange);

        position = Nd4j.create(distribution.sample(dimensions), new int[]{dimensions});
    }

    /**
     * Update the personal best if the current evaluation is better.
     */
    public void updatePersonalBest() {
        double eval = function.evaluate(position);
        if (eval < bestEval) {
            bestPosition = position.dup();
            bestEval = eval;
        }
    }

    // ==================================================================
    // Particle Kinematics
    /**
     * Get a copy of the position of the particle.
     *
     * @return the x position
     */
    public INDArray getPosition() {
        return position.dup();
    }

    /**
     * Get a copy of the velocity of the particle.
     *
     * @return the velocity
     */
    public INDArray getVelocity() {
        return velocity.dup();
    }

    /**
     * Get a copy of the personal best solution.
     *
     * @return the best position
     */
    public INDArray getBestPosition() {
        return bestPosition.dup();
    }

    /**
     * Get the value of the personal best solution.
     *
     * @return the evaluation
     */
    public double getBestEval() {
        return bestEval;
    }

    /**
     * Update the position of a particle by adding its velocity to its position.
     */
    public void updatePosition() {
        this.position.add(velocity);
    }

    /**
     * Set the velocity of the particle.
     *
     * @param velocity the new velocity
     */
    public void setVelocity(INDArray velocity) {
        this.velocity = velocity.dup();
    }

    // ==================================================================
    public void setRand(RandomGenerator rand) {
        this.rand = rand;
    }

    /**
     *
     * @param lowerRange the minimum xyz values of the position (inclusive)
     * @param upperRange the maximum xyz values of the position (exclusive)
     */
    public void setUpperLower(double upperRange, double lowerRange) {
        if (lowerRange >= upperRange) {
            throw new IllegalArgumentException("Begin range must be less than end range.");
        }
        this.upperRange = upperRange;
        this.lowerRange = lowerRange;
    }
}
