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
package fit.astro.vsa.common.utilities.math.relational;

import java.util.ArrayList;
import java.util.List;
import org.apache.commons.math3.analysis.function.Sqrt;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealVector;
import org.junit.After;
import org.junit.AfterClass;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;
import static org.junit.Assert.*;

/**
 *
 * @author Kyle Johnston <kyjohnst2000@my.fit.edu>
 */
public class RelationalOperationsTest {

    public RelationalOperationsTest() {
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

    private static final double EPS = 1.0e-5;
    
    private static final RealVector IN_VECTOR_A
            = MatrixUtils.createRealVector(new double[]{4.0, 2.0, 3.0});

    private static final RealVector IN_VECTOR_B
            = MatrixUtils.createRealVector(new double[]{3.0, 5.0, -1.0});

    
    @Test
    public void testRemoveRelationOperations() {
        
        // Who are we selecting
        List<Boolean> logicOutput = RelationalOperations
                .compareVector2Pt(IN_VECTOR_A, 3,
                        RelationalOperators.LTEQ);

        // What do we want to do to the reduced set
        RealVector update = RelationalOperations
                .removeRelationOperations(
                        IN_VECTOR_A, logicOutput);

        assertEquals(2.0, update.getEntry(0), EPS);
        assertEquals(3.0, update.getEntry(1), EPS);

    }
    
    @Test
    public void testApplyRelations() {
        
        // Who are we selecting
        List<Boolean> logicOutput = RelationalOperations
                .compareVector2Pt(IN_VECTOR_A, 3,
                        RelationalOperators.LTEQ);

        // What do we want to do to the reduced set
        RealVector update = RelationalOperations
                .applyRelationOperations(new Sqrt(),
                        IN_VECTOR_A, logicOutput);

        assertEquals(4.0, update.getEntry(0), EPS);
        assertEquals(Math.sqrt(2.0), update.getEntry(1), EPS);
        assertEquals(Math.sqrt(3.0), update.getEntry(2), EPS);

    }

    @Test
    public void testCompareVector2Pt() {

        List<Boolean> logicOutput = RelationalOperations
                .compareVector2Pt(IN_VECTOR_A, 3,
                        RelationalOperators.LTEQ);

        assertEquals(Boolean.FALSE, logicOutput.get(0));
        assertEquals(Boolean.TRUE, logicOutput.get(1));
        assertEquals(Boolean.TRUE, logicOutput.get(2));

        // =========================================
        logicOutput = RelationalOperations
                .compareVector2Pt(IN_VECTOR_A, 3,
                        RelationalOperators.GTEQ);

        assertEquals(Boolean.TRUE, logicOutput.get(0));
        assertEquals(Boolean.FALSE, logicOutput.get(1));
        assertEquals(Boolean.TRUE, logicOutput.get(2));

        // =========================================
        logicOutput = RelationalOperations
                .compareVector2Pt(IN_VECTOR_A, 3,
                        RelationalOperators.EQ);

        assertEquals(Boolean.FALSE, logicOutput.get(0));
        assertEquals(Boolean.FALSE, logicOutput.get(1));
        assertEquals(Boolean.TRUE, logicOutput.get(2));

        // =========================================
        logicOutput = RelationalOperations
                .compareVector2Pt(IN_VECTOR_A, 3,
                        RelationalOperators.GT);

        assertEquals(Boolean.TRUE, logicOutput.get(0));
        assertEquals(Boolean.FALSE, logicOutput.get(1));
        assertEquals(Boolean.FALSE, logicOutput.get(2));

        // =========================================
        logicOutput = RelationalOperations
                .compareVector2Pt(IN_VECTOR_A, 3,
                        RelationalOperators.LT);

        assertEquals(Boolean.FALSE, logicOutput.get(0));
        assertEquals(Boolean.TRUE, logicOutput.get(1));
        assertEquals(Boolean.FALSE, logicOutput.get(2));

        // =========================================
        logicOutput = RelationalOperations
                .compareVector2Pt(IN_VECTOR_A, 3,
                        RelationalOperators.NEQ);

        assertEquals(Boolean.TRUE, logicOutput.get(0));
        assertEquals(Boolean.TRUE, logicOutput.get(1));
        assertEquals(Boolean.FALSE, logicOutput.get(2));

    }

    @Test
    public void testCompareVector2Vector() {
        List<Boolean> logicOutput = RelationalOperations
                .compareVector2Vector(IN_VECTOR_A, IN_VECTOR_B,
                        RelationalOperators.LTEQ);

        assertEquals(Boolean.FALSE, logicOutput.get(0));
        assertEquals(Boolean.TRUE, logicOutput.get(1));
        assertEquals(Boolean.FALSE, logicOutput.get(2));

        // =========================================
        logicOutput = RelationalOperations
                .compareVector2Vector(IN_VECTOR_A, IN_VECTOR_B,
                        RelationalOperators.GTEQ);

        assertEquals(Boolean.TRUE, logicOutput.get(0));
        assertEquals(Boolean.FALSE, logicOutput.get(1));
        assertEquals(Boolean.TRUE, logicOutput.get(2));

        // =========================================
        logicOutput = RelationalOperations
                .compareVector2Vector(IN_VECTOR_A, IN_VECTOR_B,
                        RelationalOperators.EQ);

        assertEquals(Boolean.FALSE, logicOutput.get(0));
        assertEquals(Boolean.FALSE, logicOutput.get(1));
        assertEquals(Boolean.FALSE, logicOutput.get(2));

        // =========================================
        logicOutput = RelationalOperations
                .compareVector2Vector(IN_VECTOR_A, IN_VECTOR_B,
                        RelationalOperators.GT);

        assertEquals(Boolean.TRUE, logicOutput.get(0));
        assertEquals(Boolean.FALSE, logicOutput.get(1));
        assertEquals(Boolean.TRUE, logicOutput.get(2));

        // =========================================
        logicOutput = RelationalOperations
                .compareVector2Vector(IN_VECTOR_A, IN_VECTOR_B,
                        RelationalOperators.LT);

        assertEquals(Boolean.FALSE, logicOutput.get(0));
        assertEquals(Boolean.TRUE, logicOutput.get(1));
        assertEquals(Boolean.FALSE, logicOutput.get(2));

        // =========================================
        logicOutput = RelationalOperations
                .compareVector2Vector(IN_VECTOR_A, IN_VECTOR_B,
                        RelationalOperators.NEQ);

        assertEquals(Boolean.TRUE, logicOutput.get(0));
        assertEquals(Boolean.TRUE, logicOutput.get(1));
        assertEquals(Boolean.TRUE, logicOutput.get(2));
    }

    @Test
    public void testCompareList2Pt() {

        List<Double> doubleListA = new ArrayList<>();
        for (int idx = 0; idx < IN_VECTOR_A.getDimension(); idx++) {
            doubleListA.add(IN_VECTOR_A.getEntry(idx));
        }

        // =========================================
        List<Boolean> logicOutput = RelationalOperations
                .compareList2Pt(doubleListA, 3,
                        RelationalOperators.LTEQ);

        assertEquals(Boolean.FALSE, logicOutput.get(0));
        assertEquals(Boolean.TRUE, logicOutput.get(1));
        assertEquals(Boolean.TRUE, logicOutput.get(2));

        // =========================================
        logicOutput = RelationalOperations
                .compareList2Pt(doubleListA, 3,
                        RelationalOperators.GTEQ);

        assertEquals(Boolean.TRUE, logicOutput.get(0));
        assertEquals(Boolean.FALSE, logicOutput.get(1));
        assertEquals(Boolean.TRUE, logicOutput.get(2));

        // =========================================
        logicOutput = RelationalOperations
                .compareList2Pt(doubleListA, 3,
                        RelationalOperators.EQ);

        assertEquals(Boolean.FALSE, logicOutput.get(0));
        assertEquals(Boolean.FALSE, logicOutput.get(1));
        assertEquals(Boolean.TRUE, logicOutput.get(2));

        // =========================================
        logicOutput = RelationalOperations
                .compareList2Pt(doubleListA, 3,
                        RelationalOperators.GT);

        assertEquals(Boolean.TRUE, logicOutput.get(0));
        assertEquals(Boolean.FALSE, logicOutput.get(1));
        assertEquals(Boolean.FALSE, logicOutput.get(2));

        // =========================================
        logicOutput = RelationalOperations
                .compareList2Pt(doubleListA, 3,
                        RelationalOperators.LT);

        assertEquals(Boolean.FALSE, logicOutput.get(0));
        assertEquals(Boolean.TRUE, logicOutput.get(1));
        assertEquals(Boolean.FALSE, logicOutput.get(2));

        // =========================================
        logicOutput = RelationalOperations
                .compareList2Pt(doubleListA, 3,
                        RelationalOperators.NEQ);

        assertEquals(Boolean.TRUE, logicOutput.get(0));
        assertEquals(Boolean.TRUE, logicOutput.get(1));
        assertEquals(Boolean.FALSE, logicOutput.get(2));
    }

    @Test
    public void testCompareList2List() {

        List<Double> doubleListA = new ArrayList<>();
        for (int idx = 0; idx < IN_VECTOR_A.getDimension(); idx++) {
            doubleListA.add(IN_VECTOR_A.getEntry(idx));
        }

        List<Double> doubleListB = new ArrayList<>();
        for (int idx = 0; idx < IN_VECTOR_B.getDimension(); idx++) {
            doubleListB.add(IN_VECTOR_B.getEntry(idx));
        }

        // =========================================
        List<Boolean> logicOutput = RelationalOperations
                .compareList2List(doubleListA, doubleListB,
                        RelationalOperators.LTEQ);

        assertEquals(Boolean.FALSE, logicOutput.get(0));
        assertEquals(Boolean.TRUE, logicOutput.get(1));
        assertEquals(Boolean.FALSE, logicOutput.get(2));

        // =========================================
        logicOutput = RelationalOperations
                .compareList2List(doubleListA, doubleListB,
                        RelationalOperators.GTEQ);

        assertEquals(Boolean.TRUE, logicOutput.get(0));
        assertEquals(Boolean.FALSE, logicOutput.get(1));
        assertEquals(Boolean.TRUE, logicOutput.get(2));

        // =========================================
        logicOutput = RelationalOperations
                .compareList2List(doubleListA, doubleListB,
                        RelationalOperators.EQ);

        assertEquals(Boolean.FALSE, logicOutput.get(0));
        assertEquals(Boolean.FALSE, logicOutput.get(1));
        assertEquals(Boolean.FALSE, logicOutput.get(2));

        // =========================================
        logicOutput = RelationalOperations
                .compareList2List(doubleListA, doubleListB,
                        RelationalOperators.GT);

        assertEquals(Boolean.TRUE, logicOutput.get(0));
        assertEquals(Boolean.FALSE, logicOutput.get(1));
        assertEquals(Boolean.TRUE, logicOutput.get(2));

        // =========================================
        logicOutput = RelationalOperations
                .compareList2List(doubleListA, doubleListB,
                        RelationalOperators.LT);

        assertEquals(Boolean.FALSE, logicOutput.get(0));
        assertEquals(Boolean.TRUE, logicOutput.get(1));
        assertEquals(Boolean.FALSE, logicOutput.get(2));

        // =========================================
        logicOutput = RelationalOperations
                .compareList2List(doubleListA, doubleListB,
                        RelationalOperators.NEQ);

        assertEquals(Boolean.TRUE, logicOutput.get(0));
        assertEquals(Boolean.TRUE, logicOutput.get(1));
        assertEquals(Boolean.TRUE, logicOutput.get(2));
    }

}
