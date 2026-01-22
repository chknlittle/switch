#!/bin/bash
# View bridge logs

journalctl --user -u xmpp-opencode-bridge.service -f
