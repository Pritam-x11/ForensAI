import { useRef, useState } from 'react'

const BACKEND_URL = 'http://localhost:8080'

function App() {
  const fileInputRef = useRef(null)

  /*
   * LOGIN STATE
   */

  const [isLoggedIn, setIsLoggedIn] = useState(
    !!localStorage.getItem('forensai_token')
  )

  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [loginError, setLoginError] = useState('')
  const [loginLoading, setLoginLoading] = useState(false)

  /*
   * EVIDENCE STATE
   */

  const [selectedFile, setSelectedFile] = useState(null)
  const [caseReference, setCaseReference] = useState('')
  const [evidenceType, setEvidenceType] = useState('')
  const [matchingResults, setMatchingResults] = useState([])
  const [feedbackStatus, setFeedbackStatus] = useState({})

  /*
   * LOGIN
   */

  const handleLogin = async (event) => {
    event.preventDefault()

    setLoginError('')

    if (!username.trim()) {
      setLoginError('Please enter username')
      return
    }

    if (!password) {
      setLoginError('Please enter password')
      return
    }

    setLoginLoading(true)

    try {
      const response = await fetch(
        `${BACKEND_URL}/api/auth/login`,
        {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            username: username,
            password: password,
          }),
        }
      )

      if (!response.ok) {
        throw new Error(
          'Invalid username or password'
        )
      }

      /*
       * Backend directly returns JWT
       * as plain text.
       */
      const token = await response.text()

      console.log('Login successful')
      console.log('JWT token received')

      if (!token || !token.trim()) {
        throw new Error(
          'JWT token not received from backend'
        )
      }

      localStorage.setItem(
        'forensai_token',
        token.trim()
      )

      setIsLoggedIn(true)

      setUsername('')
      setPassword('')
      setLoginError('')

    } catch (error) {
      console.error(
        'Login error:',
        error
      )

      setLoginError(
        error.message || 'Login failed'
      )
    } finally {
      setLoginLoading(false)
    }
  }

  /*
   * LOGOUT
   */

  const handleLogout = () => {
    localStorage.removeItem(
      'forensai_token'
    )

    setIsLoggedIn(false)

    setSelectedFile(null)
    setCaseReference('')
    setEvidenceType('')
    setMatchingResults([])
    setFeedbackStatus({})
  }

  /*
   * UPLOAD BUTTON
   */

  const handleUploadClick = () => {
    if (!caseReference.trim()) {
      alert('Please enter Case Reference')
      return
    }

    if (!evidenceType) {
      alert('Please select Evidence Type')
      return
    }

    fileInputRef.current.click()
  }

  /*
   * FILE UPLOAD
   */

  const handleFileChange = async (event) => {
    const file = event.target.files[0]

    if (!file) {
      return
    }

    setSelectedFile(file)

    const formData = new FormData()

    formData.append(
      'file',
      file
    )

    formData.append(
      'caseReference',
      caseReference
    )

    formData.append(
      'evidenceType',
      evidenceType
    )

    try {
      const token =
        localStorage.getItem(
          'forensai_token'
        )

      const response = await fetch(
        `${BACKEND_URL}/api/evidence/upload`,
        {
          method: 'POST',

          headers: {
            Authorization:
              `Bearer ${token}`,
          },

          body: formData,
        }
      )

      if (!response.ok) {
        throw new Error(
          'Upload failed'
        )
      }

      const data =
        await response.json()

      console.log(
        'Evidence uploaded successfully:',
        data
      )

      const evidenceId =
        data.id

      if (evidenceId) {

        const resultResponse =
          await fetch(
            `${BACKEND_URL}/api/matching-results/evidence/${evidenceId}`,
            {
              headers: {
                Authorization:
                  `Bearer ${token}`,
              },
            }
          )

        if (!resultResponse.ok) {
          throw new Error(
            'Failed to fetch matching results'
          )
        }

        const results =
          await resultResponse.json()

        console.log(
          'Matching results:',
          results
        )

        setMatchingResults(
          results
        )

        setFeedbackStatus({})
      }

      alert(
        'Evidence uploaded successfully'
      )

    } catch (error) {

      console.error(
        'Upload error:',
        error
      )

      alert(
        'Evidence upload failed'
      )
    }
  }

  /*
   * INVESTIGATOR FEEDBACK
   */

  const handleFeedback = async (
    candidateIndex
  ) => {

    try {

      const token =
        localStorage.getItem(
          'forensai_token'
        )

      /*
       * STEP 1:
       * Submit investigator feedback
       */

      const feedbackResponse =
        await fetch(
          `${BACKEND_URL}/api/investigator/feedback?candidate_index=${candidateIndex}&feedback=Relevant%20Candidate`,
          {
            method: 'POST',

            headers: {
              Authorization:
                `Bearer ${token}`,
            },
          }
        )

      if (!feedbackResponse.ok) {
        throw new Error(
          'Feedback submission failed'
        )
      }

      const feedbackData =
        await feedbackResponse.json()

      console.log(
        'Investigator feedback response:',
        feedbackData
      )

      /*
       * STEP 2:
       * Prepare current candidates
       * for refined search
       */

      const candidates =
        matchingResults.map(
          (result) => ({
            candidate_index:
              result.candidateId,

            filename:
              result.candidateName,

            final_score:
              result.finalScore,
          })
        )

      const refinedSearchRequest = {
        candidates:
          candidates,

        feedback_candidate_index:
          candidateIndex,
      }

      console.log(
        'Refined search request:',
        refinedSearchRequest
      )

      /*
       * STEP 3:
       * Call refined search
       */

      const refinedResponse =
        await fetch(
          `${BACKEND_URL}/api/investigator/refined-search`,
          {
            method: 'POST',

            headers: {
              'Content-Type':
                'application/json',

              Authorization:
                `Bearer ${token}`,
            },

            body:
              JSON.stringify(
                refinedSearchRequest
              ),
          }
        )

      if (!refinedResponse.ok) {
        throw new Error(
          'Refined search failed'
        )
      }

      const refinedData =
        await refinedResponse.json()

      console.log(
        'Refined search response:',
        refinedData
      )

      /*
       * STEP 4:
       * Merge refined candidates
       * with existing results
       */

      const refinedResults =
        refinedData.candidates
          .map(
            (refinedCandidate) => {

              const originalResult =
                matchingResults.find(
                  (result) =>
                    result.candidateId ===
                    refinedCandidate.candidate_index
                )

              if (!originalResult) {
                return null
              }

              return {
                ...originalResult,

                finalScore:
                  refinedCandidate.final_score,

                feedbackRelevant:
                  refinedCandidate.feedback_relevant,
              }
            }
          )
          .filter(
            (result) =>
              result !== null
          )

      /*
       * STEP 5:
       * Update frontend table
       */

      setMatchingResults(
        refinedResults
      )

      /*
       * STEP 6:
       * Show feedback status
       */

      setFeedbackStatus(
        (previousStatus) => ({
          ...previousStatus,

          [candidateIndex]:
            'Marked Relevant',
        })
      )

      console.log(
        'Frontend results refined successfully'
      )

    } catch (error) {

      console.error(
        'Investigator feedback/refined search error:',
        error
      )

      alert(
        'Failed to process investigator feedback'
      )
    }
  }

  /*
   * LOGIN PAGE
   */

  if (!isLoggedIn) {

    return (
      <div className="min-h-screen bg-slate-950 text-white flex items-center justify-center px-4">

        <div className="w-full max-w-md">

          <div className="text-center mb-8">

            <h1 className="text-5xl font-bold text-cyan-400">
              ForensAI
            </h1>

            <p className="mt-4 text-slate-300">
              AI-Powered Multimodal Forensic
              Identification Support System
            </p>

          </div>

          <div className="rounded-2xl border border-slate-700 bg-slate-900 p-8 shadow-2xl">

            <h2 className="text-2xl font-bold text-white text-center">
              Investigator Login
            </h2>

            <p className="mt-2 text-center text-slate-400 text-sm">
              Sign in to access the ForensAI evidence system
            </p>

            <form
              onSubmit={handleLogin}
              className="mt-8"
            >

              <label className="block text-sm font-medium text-slate-300">
                Username
              </label>

              <input
                type="text"
                placeholder="Enter username"
                value={username}
                onChange={(event) =>
                  setUsername(
                    event.target.value
                  )
                }
                className="mt-2 w-full rounded-lg border border-slate-700 bg-slate-950 px-4 py-3 text-white outline-none focus:border-cyan-400"
              />

              <label className="mt-5 block text-sm font-medium text-slate-300">
                Password
              </label>

              <input
                type="password"
                placeholder="Enter password"
                value={password}
                onChange={(event) =>
                  setPassword(
                    event.target.value
                  )
                }
                className="mt-2 w-full rounded-lg border border-slate-700 bg-slate-950 px-4 py-3 text-white outline-none focus:border-cyan-400"
              />

              {loginError && (
                <div className="mt-4 rounded-lg border border-red-500/30 bg-red-500/10 px-4 py-3 text-sm text-red-400">
                  {loginError}
                </div>
              )}

              <button
                type="submit"
                disabled={loginLoading}
                className="mt-6 w-full rounded-lg bg-cyan-500 px-6 py-3 font-semibold text-slate-950 transition hover:bg-cyan-400 disabled:cursor-not-allowed disabled:opacity-50"
              >
                {loginLoading
                  ? 'Signing in...'
                  : 'Login'}
              </button>

            </form>

          </div>

        </div>

      </div>
    )
  }

  /*
   * EVIDENCE PAGE
   */

  return (
    <div className="min-h-screen bg-slate-950 text-white px-6 py-10">

      <div className="mx-auto max-w-6xl">

        {/* HEADER */}

        <div className="flex items-center justify-between">

          <div>

            <h1 className="text-5xl font-bold text-cyan-400">
              ForensAI
            </h1>

            <p className="mt-3 text-xl text-slate-300">
              AI-Powered Multimodal Forensic Identification Support System
            </p>

          </div>

          <button
            onClick={handleLogout}
            className="rounded-lg border border-red-500/50 bg-red-500/10 px-5 py-3 font-semibold text-red-400 hover:bg-red-500/20"
          >
            Logout
          </button>

        </div>

        {/* EVIDENCE UPLOAD */}

        <div className="mt-10 rounded-2xl border border-slate-700 bg-slate-900 p-6">

          <h2 className="text-2xl font-bold text-cyan-400">
            Evidence Upload
          </h2>

          <div className="mt-6 grid grid-cols-1 gap-4 md:grid-cols-2">

            <input
              type="text"
              placeholder="Enter Case Reference"
              value={caseReference}
              onChange={(event) =>
                setCaseReference(
                  event.target.value
                )
              }
              className="w-full rounded-lg border border-slate-700 bg-slate-950 px-4 py-3 text-white outline-none focus:border-cyan-400"
            />

            <select
              value={evidenceType}
              onChange={(event) =>
                setEvidenceType(
                  event.target.value
                )
              }
              className="w-full rounded-lg border border-slate-700 bg-slate-950 px-4 py-3 text-white outline-none focus:border-cyan-400"
            >

              <option value="">
                Select Evidence Type
              </option>

              <option value="PHOTO">
                Photo
              </option>

              <option value="VIDEO">
                Video
              </option>

            </select>

          </div>

          <input
            ref={fileInputRef}
            type="file"
            accept="image/*,video/*"
            className="hidden"
            onChange={handleFileChange}
          />

          <button
            onClick={handleUploadClick}
            className="mt-6 rounded-lg bg-cyan-500 px-6 py-3 font-semibold text-slate-950 hover:bg-cyan-400"
          >
            Upload Evidence
          </button>

          {selectedFile && (
            <p className="mt-4 text-slate-300">
              Selected file: {selectedFile.name}
            </p>
          )}

        </div>

        {/* MATCHING RESULTS */}

        {matchingResults.length > 0 && (

          <div className="mt-10">

            <h2 className="mb-4 text-2xl font-bold text-cyan-400">
              Matching Results
            </h2>

            <div className="overflow-x-auto rounded-lg border border-slate-700">

              <table className="w-full text-sm">

                <thead className="bg-slate-800">

                  <tr>

                    <th className="px-4 py-3 text-left">
                      Candidate
                    </th>

                    <th className="px-4 py-3 text-left">
                      Distance
                    </th>

                    <th className="px-4 py-3 text-left">
                      Similarity
                    </th>

                    <th className="px-4 py-3 text-left">
                      Quality
                    </th>

                    <th className="px-4 py-3 text-left">
                      Final Score
                    </th>

                    <th className="px-4 py-3 text-left">
                      Investigator Feedback
                    </th>

                  </tr>

                </thead>

                <tbody>

                  {matchingResults.map(
                    (result) => (

                      <tr
                        key={result.id}
                        className="border-t border-slate-700"
                      >

                        <td className="px-4 py-3">
                          {result.candidateName}
                        </td>

                        <td className="px-4 py-3">
                          {result.distance.toFixed(2)}
                        </td>

                        <td className="px-4 py-3">
                          {result.similarityScore.toFixed(2)}
                        </td>

                        <td className="px-4 py-3">
                          {result.qualityScore.toFixed(2)}
                        </td>

                        <td className="px-4 py-3 font-semibold text-cyan-400">
                          {result.finalScore.toFixed(2)}
                        </td>

                        <td className="px-4 py-3">

                          <button
                            onClick={() =>
                              handleFeedback(
                                result.candidateId
                              )
                            }
                            className="rounded-lg bg-emerald-500 px-4 py-2 font-semibold text-slate-950 hover:bg-emerald-400"
                          >
                            Relevant
                          </button>

                          {feedbackStatus[
                            result.candidateId
                          ] && (

                            <span className="ml-3 text-emerald-400">

                              {
                                feedbackStatus[
                                  result.candidateId
                                ]
                              }

                            </span>

                          )}

                        </td>

                      </tr>

                    )
                  )}

                </tbody>

              </table>

            </div>

          </div>

        )}

      </div>

    </div>
  )
}

export default App