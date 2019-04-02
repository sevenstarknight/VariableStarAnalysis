/*
 * Copyright (C) 2018 kjohnston
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
package fit.astro.vsa.analysis.other;

import fit.astro.vsa.analysis.linear.*;
import fit.astro.vsa.analysis.linear.RawKnn_LINEAR_Vector;
import fit.astro.vsa.common.utilities.math.handling.exceptions.NotEnoughDataException;
import java.io.IOException;
import org.junit.After;
import org.junit.AfterClass;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;

/**
 *
 * @author kjohnston
 */
public class LINEAR_Raw_Test {

    private String fileLocation = "/Users/kjohnston/Google Drive/VarStarData/LinearData/LinearData.mat";

    public LINEAR_Raw_Test() {
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
    public void testLINEAR_Vector() throws IOException, NotEnoughDataException {

        RawKnn_LINEAR_Vector.execute(3, true, fileLocation);

    }

    @Test
    public void testLINEAR_Matrix() throws IOException, NotEnoughDataException {

        RawKnn_LINEAR_Matrix.execute(3, true, fileLocation);

    }

    @Test
    public void testLINEAR_PMML() throws IOException, NotEnoughDataException {

        ProcessLINEARDataViaPMML.execute(0.25, 5.0, 0.5, true, fileLocation);

    }

    @Test
    public void testLINEAR_kNN() throws IOException, NotEnoughDataException {

        ProcessLINEARDataViakNN.execute(true, fileLocation);
    }

    @Test
    public void testLINEAR_NCA() throws IOException, NotEnoughDataException {

        ProcessLINEARDataViaNCA.execute(true, fileLocation);
    }

    @Test
    public void testLINEAR_RF() throws IOException, NotEnoughDataException {

        RawRF_LINEAR_Vector.execute(0.002, true, fileLocation);
    }

}
