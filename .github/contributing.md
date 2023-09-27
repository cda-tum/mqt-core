# Contributing

Thank you for your interest in contributing to this project.
We value contributions from people with all levels of experience.
In particular if this is your first pull request not everything has to be perfect.
We will guide you through the process.

We use GitHub to [host code](https://github.com/cda-tum/mqt-core), to [track issues and feature requests](https://github.com/cda-tum/mqt-core/issues), as well as accept [pull requests](https://github.com/cda-tum/mqt-core/pulls).
See <https://docs.github.com/en/get-started/quickstart> for a general introduction to working with GitHub and contributing to projects.

## Types of Contributions

You can contribute in several ways:

- 🐛 Report Bugs
  : Report bugs at <https://github.com/cda-tum/mqt-core/issues> using the _🐛 Bug report_ issue template. Please make sure to fill out all relevant information in the respective issue form.

- 🐛 Fix Bugs
  : Look through the [GitHub Issues](https://github.com/cda-tum/mqt-core/issues) for bugs. Anything tagged with "bug" is open to whoever wants to try and fix it.

- ✨ Propose New Features
  : Propose new features at <https://github.com/cda-tum/mqt-core/issues> using the _✨ Feature request_ issue template. Please make sure to fill out all relevant information in the respective issue form.

- ✨ Implement New Features
  : Look through the [GitHub Issues](https://github.com/cda-tum/mqt-core/issues) for features. Anything tagged with "feature" or "enhancement" is open to whoever wants to implement it. We highly appreciate external contributions to the project.

- 📝 Write Documentation
  : MQT Core could always use some more documentation, and we appreciate any help with that.

## Get Started 🎉

Ready to contribute? Check out the {doc}`Development Guide <DevelopmentGuide>` to set up MQT Core for local development and learn about the style guidelines and conventions used throughout the project.

We value contributions from people with all levels of experience.
In particular if this is your first PR not everything has to be perfect.
We will guide you through the PR process.
Nevertheless, please try to follow the guidelines below as well as you can to help make the PR process quick and smooth.

## Core Guidelines

- ["Commit early and push often"](https://www.worklytics.co/blog/commit-early-push-often).
- Write meaningful commit messages (preferably using [gitmoji](https://gitmoji.dev) to give additional context to your commits).
- Focus on a single feature/bug at a time and only touch relevant files. Split multiple features into multiple contributions.
- If you added a new feature, you should add tests that ensure it works as intended. Furthermore, the new feature should be documented appropriately.
- If you fixed a bug, you should add tests that demonstrate that the bug has been fixed.
- Document your code thoroughly and write readable code.
- Keep your code clean. Remove any debug statements, left-over comments, or code unrelated to your contribution.
- Run `pre-commit run -a` to check your code for style and linting errors before committing.

## Pull Request Workflow

- Create PRs early. It is ok to create work-in-progress PRs. You may mark these as draft PRs on GitHub.
- Describe your PR. Start with a descriptive title, reference any related issues by including the issue number in the PR description, and add a comprehensive description of the changes. We provide a PR template that you can (and should) follow to create a PR. Do not delete any sections from the template.
- Whenever a PR is created or updated, several workflows on all supported platforms and versions of Python are executed. These workflows ensure that the project still builds, that all tests pass, and that the code is properly formatted and introduces no new linting errors. Your PR is expected to pass all these continuous integration (CI) checks before it can be merged. Here are some tips for finding the cause of certain failures:
  - If any of the `CI / 🇨‌ Test` checks fail, this most likely indicates build errors or test failures in the C++ part of the code base. Look through the respective logs on GitHub for any error or failure messages.
  - If any of the `CI / 🐍 Test` checks fail, this most likely indicates build errors or test failures in the Python part of the code base. Look through the respective logs on GitHub for any error or failure messages.
  - If any of the `codecov/\*` checks fail, this means that your changes are not appropriately covered by tests or that the overall project coverage decreased too much. Ensure that you include tests for all your changes in the PR.
  - If `cpp-linter` comments on your PR with a list of warnings, these have been raised by `clang-tidy` when checking the C++ part of your changes for warnings or style guideline violations. The individual messages frequently provide helpful suggestions on how to fix the warnings. If you don't see any messages, but the `🇨‌ Lint / 🚨 Lint` check is red, click on the `Details` link to see the full log of the check and a step summary.
  - If the `pre-commit.ci` check fails, some of the `pre-commit` checks failed and could not be fixed automatically by the _pre-commit.ci_ bot. Such failures are most likely related to the Python part of the code base. The individual log messages frequently provide helpful suggestions on how to fix the warnings.
  - If the `docs/readthedocs.org:mqt-core` check fails, the documentation could not be built properly. Inspect the corresponding log file for any errors.
- Once your PR is ready, change it from a draft PR to a regular PR and request a review from one of the project maintainers. Please make sure to only request a review once you are done with your changes and the PR is ready to be merged. If you are unsure whether your PR is ready for review, please ask in the PR comments.
- If your PR gets a "Changes requested" review, you will need to address the feedback and update your PR by pushing to the same branch. You don't need to close the PR and open a new one. Respond to review comments on the PR (e.g., with "done 👍" or "done in @\<commit\>") to let the reviewer know that you have addressed the feedback. Note that reviewers do not get a notification if you just react to the review comment with an emoji. You need to write a comment to notify the reviewer.
- Be sure to re-request review once you have made changes after a code review so that maintainers know that the requests have been addressed.

---

This document was inspired by and partially adapted from

- <https://matplotlib.org/stable/devel/coding_guide.html>
- <https://opensource.creativecommons.org/contributing-code/pr-guidelines/>
- <https://yeoman.io/contributing/pull-request.html>
- <https://github.com/scikit-build/scikit-build>
