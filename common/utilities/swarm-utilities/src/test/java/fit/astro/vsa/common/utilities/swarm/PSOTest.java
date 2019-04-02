package fit.astro.vsa.common.utilities.swarm;

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

import org.apache.commons.math3.random.MersenneTwister;
import org.apache.commons.math3.random.RandomGenerator;
import org.junit.After;
import org.junit.AfterClass;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;
import static org.junit.Assert.*;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 *
 * @author kjohnston
 */
public class PSOTest {

    private RandomGenerator rand = new MersenneTwister(42L);

    public PSOTest() {
    }

    @BeforeClass
    public static void setUpClass() {
    }

    @AfterClass
    public static void tearDownClass() {
    }

    @Before
    public void setUp() {
    }

    @After
    public void tearDown() {
    }

    @Test
    public void testPSOSwarm_Ackley() {

        // Domain is [-5, 5] Minimum is 0 at x = 0.0 & y = 0.0
        SwarmOptimizationFunction function = SwarmTestFunctions.ackleysFunction();

        int particles = 1000, epochs = 1000;

        ParticleSwarmOptimization swarm
                = new ParticleSwarmOptimization(function, 2, particles, epochs);

        swarm.setLowerRange(-5.0);
        swarm.setUpperRange(5.0);
        swarm.setRand(rand);

        INDArray bestEstimate = swarm.run();

        assertEquals(0.0, bestEstimate.getDouble(0), 1e-3);
        assertEquals(0.0, bestEstimate.getDouble(1), 1e-3);

        assertEquals(0.0, function.evaluate(bestEstimate), 1e-3);

    }

    @Test
    public void testPSOSwarm_Booth() {

        // Domain is [-5, 5] Minimum is 0 at x = 0.0 & y = 0.0
        SwarmOptimizationFunction function = SwarmTestFunctions.boothsFunction();

        int particles = 1000, epochs = 1000;

        ParticleSwarmOptimization swarm
                = new ParticleSwarmOptimization(function, 2, particles, epochs);

        swarm.setLowerRange(-5.0);
        swarm.setUpperRange(5.0);
        swarm.setRand(rand);

        INDArray bestEstimate = swarm.run();

        assertEquals(1.0, bestEstimate.getDouble(0), 1e-3);
        assertEquals(3.0, bestEstimate.getDouble(1), 1e-3);

        assertEquals(0.0, function.evaluate(bestEstimate), 1e-3);

    }
}
