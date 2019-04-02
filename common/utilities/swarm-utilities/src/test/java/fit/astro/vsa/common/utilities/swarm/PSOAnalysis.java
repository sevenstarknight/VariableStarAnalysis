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

import java.util.Scanner;

/**
 * @author LazoCoder, from https://github.com/LazoCoder
 * @author kjohnston, Updates, clean up, adjustments to Java8
 */
public class PSOAnalysis {

    public static void main(String[] args) {

        SwarmOptimizationFunction function = SwarmTestFunctions.ackleysFunction();

        int particles = 1000, epochs = 1000;

        ParticleSwarmOptimization swarm
                = new ParticleSwarmOptimization(function, 2, particles, epochs);

        
        swarm.run();
    }

}
